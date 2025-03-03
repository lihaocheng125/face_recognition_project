# face_recognition_project

## 项目概述
这是一个基于Django框架开发的人脸识别系统。该系统允许用户进行注册和登录，通过摄像头捕获人脸图像，并使用人脸识别技术进行身份验证。

## 主要功能
- **用户注册**：用户可以输入用户名，系统会通过摄像头捕获用户的人脸图像，并将其编码存储到数据库中。
- **用户登录**：用户登录时，系统会通过摄像头捕获人脸图像，与人脸数据库中的编码进行比对，如果匹配成功则允许用户登录到仪表盘页面。
- **仪表盘**：用户成功登录后，会进入仪表盘页面，显示登录成功的信息。

## 项目结构
项目主要由以下几个部分组成：

### 应用程序（`face_app`）
- **`views.py`**：包含处理用户请求的视图函数，如 `index`、`register`、`login` 和 `dashboard`。
- **`models.py`**：定义了数据模型，如 `UserFace` 用于存储用户的用户名和人脸编码。
- **`templates` 文件夹**：包含HTML模板文件，如 `index.html`、`register.html`、`login.html` 和 `dashboard.html`。

### 项目配置
- **`settings.py`**：Django项目的配置文件，包括数据库配置、应用程序列表、中间件等。
- **`urls.py`**：定义了URL路由规则，将URL映射到相应的视图函数。

### 管理脚本
- **`manage.py`**：Django的命令行管理工具，用于启动开发服务器、创建数据库表等。

## 安装与运行

### 克隆项目
首先，克隆项目到本地：
```bash
git clone <项目仓库地址>
cd face_recognition_project