package xplatai

import (
	"archive/zip"
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"time"
)

const lcp_VERSION = "b6209" // commit version
const DEFAULT_HF_MODEL = "tensorblock/pygmalion-2-7b-GGUF:Q4_K_M"

type hostOs = int

const (
	_OS_WIN hostOs = iota
	_OS_MAC
	_OS_UBUNTU
)

var osNames = map[hostOs]string{
	_OS_WIN:    "win",
	_OS_MAC:    "macos",
	_OS_UBUNTU: "ubuntu",
}

type hostArch = int

const (
	_ARCH_x64 hostArch = iota
	_ARCH_ARM
)

var archNames = map[hostArch]string{
	_ARCH_x64: "x64",
	_ARCH_ARM: "arm64",
}

type HostHardware = int

const (
	HW_CPU HostHardware = iota
	HW_CUDA
	HW_VULKAN
	HW_RADEON
	HW_NONE
)

var hwNames = map[HostHardware]string{
	HW_CPU:    "cpu",
	HW_CUDA:   "cuda-12.4",
	HW_VULKAN: "vulkan",
	HW_RADEON: "hip-radeon",
	HW_NONE:   "",
}

type hostInfo struct {
	opSys    hostOs
	hardware HostHardware
	arch     hostArch
}

type XpltAI struct {
	proc   *exec.Cmd
	client *http.Client
	port   string
	isConn bool
}

func New(hfModelName string, port string) (*XpltAI, error) {
	if hfModelName == "" {
		hfModelName = DEFAULT_HF_MODEL
	}

	xai := &XpltAI{}

	xai.client = &http.Client{}
	xai.port = port

	cwd, err := os.Getwd()
	if err != nil {
		return xai, err
	}

	serverPath := path.Join(cwd, "llamacpp", "llama-server.exe")
	exists, _ := isPathExist(serverPath)
	if !exists {
		return xai, errors.New("could not find llama.cpp binaries, run DownloadRequirements to fix")
	}

	xai.proc = exec.Command(
		serverPath,
		"-hf", hfModelName,
		"--port", port,
		"--threads", "6",
	)
	xai.proc.SysProcAttr = &syscall.SysProcAttr{
		HideWindow:    false,
		CreationFlags: 0,
	}

	err = xai.proc.Start()
	if err != nil {
		return xai, err
	}
	return xai, nil
}

func (x *XpltAI) Close() error {
	return x.proc.Process.Kill()
}

func (x *XpltAI) WaitUntilLoaded(timeout time.Duration) error {
	if timeout.Milliseconds() <= 0 {
		timeout = 99 * time.Minute
	}

	timeoutTime := time.Now().Add(timeout)

	for time.Now().Before(timeoutTime) {
		resp, err := x.client.Head("http://127.0.0.1:" + x.port + "/health")
		if err == nil && resp.StatusCode < 500 {
			return nil
		}
		time.Sleep(time.Second * 1)
	}
	return errors.New("time out")
}

func (x *XpltAI) Chat(messages []map[string]string, maxTokens int) (string, error) {
	if maxTokens <= 0 {
		maxTokens = 150
	}

	if !x.isConn {
		// Test for availability
		for i := range 10 {
			time.Sleep(time.Second * time.Duration(i))
			resp, err := x.client.Head("http://127.0.0.1:" + x.port + "/health")
			if err == nil && resp.StatusCode < 500 {
				break
			}

			if i == 9 {
				return "", errors.New("timed out while waiting for llama.cpp")
			}
		}
	}

	data := map[string]any{
		"messages":   messages,
		"max_tokens": maxTokens,
		"stop": []string{
			"<|",
		},
	}

	b, err := json.Marshal(data)
	if err != nil {
		return "", err
	}
	reqBody := bytes.NewBuffer(b)

	req, err := http.NewRequest("POST", "http://127.0.0.1:"+x.port+"/v1/chat/completions", reqBody)
	if err != nil {
		return "", err
	}

	resp, err := x.client.Do(req)
	if err != nil {
		return "", err
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	// Check if json
	if len(body) > 0 && body[0] != '{' {
		return "", errors.New(string(body))
	}

	jsonData := make(map[string]any, 8)

	err = json.Unmarshal(body, &jsonData)
	if err != nil {
		return "", err
	}

	choices, ok := jsonData["choices"].([]any)
	if !ok {
		return "", errors.New("json parsing failure, missing choices field")
	}

	if len(choices) == 0 {
		return "", errors.New("json parsing failure, field choices is empty")
	}

	choice, ok := choices[0].(map[string]any)
	if !ok {
		return "", errors.New("json parsing failure, failed to get choice")
	}

	message, ok := choice["message"].(map[string]any)
	if !ok {
		return "", errors.New("json parsing failure, missing message field")
	}

	content, ok := message["content"].(string)
	if !ok {
		return "", errors.New("json parsing failure, missing content field")
	}

	x.isConn = true
	return strings.TrimSpace(content), nil
}

func (x *XpltAI) Complete(prompt string, maxTokens int) (string, error) {
	if maxTokens <= 0 {
		maxTokens = 150
	}

	if !x.isConn {
		// Test for availability
		for i := range 10 {
			time.Sleep(time.Second * time.Duration(i))
			resp, err := x.client.Head("http://127.0.0.1:" + x.port + "/health")
			if err == nil && resp.StatusCode < 500 {
				break
			}

			if i == 9 {
				return "", errors.New("timed out while waiting for llama.cpp")
			}
		}
	}

	data := map[string]any{
		"prompt":    prompt,
		"n_predict": maxTokens,
		"stop": []string{
			"<|",
		},
	}

	b, err := json.Marshal(data)
	if err != nil {
		return "", err
	}
	reqBody := bytes.NewBuffer(b)

	req, err := http.NewRequest("POST", "http://127.0.0.1:"+x.port+"/completion", reqBody)
	if err != nil {
		return "", err
	}

	resp, err := x.client.Do(req)
	if err != nil {
		return "", err
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	// Check if json
	if len(body) > 0 && body[0] != '{' {
		return "", errors.New(string(body))
	}

	jsonData := make(map[string]any, 8)

	err = json.Unmarshal(body, &jsonData)
	if err != nil {
		return "", err
	}

	content, ok := jsonData["content"].(string)
	if !ok {
		return "", errors.New("json parsing failure, missing content field")
	}

	x.isConn = true
	return strings.TrimSpace(content), nil
}

func isPathExist(entPath string) (bool, error) {
	f, err := os.Open(entPath)
	if os.IsNotExist(err) {
		return false, nil
	} else if err != nil {
		return false, err
	}
	defer f.Close()
	return true, nil
}

func isLlamCppServerExist() (bool, error) {
	cwd, err := os.Getwd()
	if err != nil {
		return false, err
	}

	serverPath := path.Join(cwd, "llamacpp", "llama-server.exe")
	return isPathExist(serverPath)
}

func IsRequDownloaded() bool {
	exists, _ := isLlamCppServerExist()
	return exists
}

func getDownloadUrl(hardware HostHardware) (string, error) {
	host := hostInfo{}

	switch runtime.GOOS {
	case "windows":
		host.opSys = _OS_WIN
	case "darwin":
		host.opSys = _OS_MAC
	case "linux":
		host.opSys = _OS_UBUNTU
	default:
		return "", errors.New("unsupported operating system")
	}

	switch runtime.GOARCH {
	case "amd64":
		host.arch = _ARCH_x64
	case "arm64":
		host.arch = _ARCH_ARM
	default:
		return "", errors.New("unsupported cpu architecture")
	}

	switch host.opSys {
	case _OS_WIN:
		if hardware == HW_NONE {
			host.hardware = HW_CPU
		} else {
			host.hardware = hardware
		}
	case _OS_MAC:
		host.hardware = HW_NONE
	case _OS_UBUNTU:
		if hardware == HW_VULKAN {
			host.hardware = HW_VULKAN
		} else {
			host.hardware = HW_NONE
		}
	}

	releaseName := "https://github.com/ggml-org/llama.cpp/releases/download/" +
		lcp_VERSION + "/llama-" + lcp_VERSION + "-bin-" +
		osNames[host.opSys] + "-"

	if host.hardware != HW_NONE {
		releaseName += hwNames[host.hardware] + "-"
	}

	releaseName += archNames[host.arch] + ".zip"
	return releaseName, nil
}

func DownloadRequirements(hardware HostHardware) error {
	url, err := getDownloadUrl(hardware)
	if err != nil {
		return err
	}

	cwd, err := os.Getwd()
	if err != nil {
		return err
	}

	dirPath := path.Join(cwd, "llamacpp")
	err = os.RemoveAll(dirPath)
	if err != nil {
		return err
	}
	err = os.MkdirAll(dirPath, os.ModeDir)
	if err != nil {
		return err
	}

	client := http.Client{}
	resp, err := client.Get(url)
	if err != nil {
		return err
	}
	if resp.StatusCode >= 300 {
		errMsg, _ := io.ReadAll(resp.Body)
		return errors.New(string(errMsg))
	}
	defer resp.Body.Close()

	zipPath := path.Join(cwd, "llamacpp", "llamacpp.zip")
	zipf, err := os.OpenFile(zipPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer zipf.Close()

	buff := make([]byte, 512)

	for {
		n, err := resp.Body.Read(buff)
		if n > 0 {
			_, err2 := zipf.Write(buff[:n])
			if err2 != nil {
				return err2
			}
		}

		if err == io.EOF {
			break
		} else if err != nil {
			return err
		}
	}
	zipf.Close()

	archive, err := zip.OpenReader(zipPath)
	if err != nil {
		return err
	}
	defer archive.Close()

	for _, f := range archive.File {
		fPath := path.Join(dirPath, f.Name)

		if f.FileInfo().IsDir() {
			os.MkdirAll(fPath, os.ModePerm)
			continue
		}

		err := os.MkdirAll(filepath.Dir(fPath), os.ModePerm)
		if err != nil {
			return err
		}

		dstFile, err := os.OpenFile(fPath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, f.Mode())
		if err != nil {
			return err
		}
		defer dstFile.Close()

		fileInArchive, err := f.Open()
		if err != nil {
			return err
		}
		defer fileInArchive.Close()

		_, err = io.Copy(dstFile, fileInArchive)
		if err != nil {
			return err
		}
	}

	return nil
}

func PreFetchModel(hfModelName string) error {
	if hfModelName == "" {
		hfModelName = DEFAULT_HF_MODEL
	}

	cwd, err := os.Getwd()
	if err != nil {
		return err
	}

	cliPath := path.Join(cwd, "llamacpp", "llama-cli.exe")
	exists, _ := isPathExist(cliPath)
	if !exists {
		return errors.New("could not find llama.cpp binaries, run DownloadRequirements to fix")
	}

	proc := exec.Command(
		cliPath,
		"-hf", hfModelName,
		"-n", "1",
		"-no-cnv",
	)

	out, err := proc.CombinedOutput()
	os.Stdout.Write(out)
	if err != nil {
		return err
	}
	return nil
}
