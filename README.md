# Introduction 
This tool uses AI to judge the results from the Crowd Intelligence Platform Judger..

## Prerequisites

- Visual Studio Code with Python 3.11 +

## Build and debug steps

To build everything with default settings run this command in repository root:
```
pip install -r requirements.txt
```

To pack:
```
pyinstaller --onedir --paths=. --name CIP_tool main.py
```

To run exe:
```
CIP_tool.exe "<CIP item detail page link>"
```
