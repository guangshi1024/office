import sys
import os
import cv2
import numpy as np
import time
import logging
import hashlib
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import torch

# 手动指定缓存路径（和报错的路径一致）
torch.hub.set_dir(r"D:\biyesheji\.cache\torch\hub")

# -------------------- PyQt5 --------------------
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QGroupBox, QTextEdit, QMessageBox, QProgressBar,
                             QSplitter, QDoubleSpinBox, QGridLayout, QLineEdit,
                             QDialog, QFormLayout, QStackedWidget, QTableWidget,
                             QTableWidgetItem, QHeaderView, QCheckBox, QComboBox,
                             QTabWidget, QMenuBar, QMenu, QAction, QStatusBar,
                             QFrame, QSpacerItem, QSizePolicy, QDesktopWidget)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex, QMutexLocker, QSize
from PyQt5.QtGui import QImage, QPixmap, QIcon, QFont, QPalette, QColor

# -------------------- 日志 --------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -------------------- YOLO 可用性 --------------------
try:
    import torch
    import torchvision

    YOLO_AVAILABLE = True
    logger.info("YOLO 依赖已加载")
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("YOLO 依赖未安装，请先安装：pip install torch torchvision ultralytics")


# =============================================================================
#  用户管理系统
# =============================================================================
class UserManager:
    """用户管理系统 - 支持管理员和普通用户"""

    def __init__(self, config_file: str = "users.json"):
        self.config_file = config_file
        self.users: Dict[str, dict] = {}
        self.current_user: Optional[str] = None
        self.is_admin: bool = False
        self.load_users()

        # 如果没有用户，创建默认管理员
        if not self.users:
            self.add_user("admin", "admin123", is_admin=True)
            logger.info("创建默认管理员账户: admin/admin123")

    def _hash_password(self, password: str) -> str:
        """MD5密码加密"""
        return hashlib.md5(password.encode()).hexdigest()

    def load_users(self):
        """加载用户配置"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.users = json.load(f)
            except Exception as e:
                logger.error(f"加载用户配置失败: {e}")
                self.users = {}

    def save_users(self):
        """保存用户配置"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.users, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存用户配置失败: {e}")

    def add_user(self, username: str, password: str, is_admin: bool = False) -> bool:
        """添加用户"""
        if username in self.users:
            return False
        self.users[username] = {
            "password": self._hash_password(password),
            "is_admin": is_admin,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "last_login": None
        }
        self.save_users()
        return True

    def delete_user(self, username: str) -> bool:
        """删除用户"""
        if username not in self.users or username == "admin":
            return False
        del self.users[username]
        self.save_users()
        return True

    def verify_user(self, username: str, password: str) -> bool:
        """验证用户"""
        if username not in self.users:
            return False
        hashed = self._hash_password(password)
        if self.users[username]["password"] == hashed:
            self.current_user = username
            self.is_admin = self.users[username]["is_admin"]
            self.users[username]["last_login"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self.save_users()
            return True
        return False

    def change_password(self, username: str, new_password: str) -> bool:
        """修改密码"""
        if username not in self.users:
            return False
        self.users[username]["password"] = self._hash_password(new_password)
        self.save_users()
        return True

    def get_user_list(self) -> List[Tuple[str, bool, str]]:
        """获取用户列表"""
        return [(name, info["is_admin"], info.get("last_login", "从未登录"))
                for name, info in self.users.items()]

    def logout(self):
        """登出"""
        self.current_user = None
        self.is_admin = False


# =============================================================================
#  YOLODetector（与原代码一致）
# =============================================================================
class YOLODetector:
    def __init__(self, model_path: str = "", device: str = "cpu", conf_threshold: float = 0.25):
        self.model_path = model_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.model = None
        self.class_names = []
        self.vehicle_classes = ['bicycle', 'car', 'motorcycle', 'airplane',
                                'bus', 'train', 'truck', 'boat']
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def load_model(self, model_path: str) -> bool:
        try:
            if not YOLO_AVAILABLE:
                logger.error("YOLO 依赖未安装");
                return False
            if not os.path.exists(model_path):
                logger.error(f"模型文件不存在: {model_path}");
                return False
            if model_path.endswith('.pt'):
                self.model = torch.hub.load('ultralytics/yolov5', 'custom',
                                            path=model_path, device=self.device)
                logger.info(f"PyTorch 模型加载成功: {model_path}")
            else:
                logger.error(f"不支持的模型格式: {model_path}");
                return False
            self.class_names = self.model.names if hasattr(self.model, 'names') else self.vehicle_classes
            self.model_path = model_path
            return True
        except Exception as e:
            logger.error(f"模型加载失败: {e}");
            return False

    def predict(self, image: np.ndarray, conf_threshold: float = None) -> List[dict]:
        if self.model is None:
            return []
        try:
            conf = conf_threshold or self.conf_threshold
            results = self.model(image)
            detections = []
            for *box, conf_, cls in results.xyxy[0]:
                if conf_ >= conf:
                    x1, y1, x2, y2 = box
                    class_id = int(cls)
                    name = self.class_names[class_id] if class_id < len(self.class_names) else str(class_id)
                    detections.append({'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                       'confidence': float(conf_),
                                       'class_id': class_id,
                                       'class_name': name})
            return detections
        except Exception as e:
            logger.error(f"推理失败: {e}");
            return []

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
        if frame is None or frame.size == 0:
            return frame, []
        detections = self.predict(frame)
        return self.draw_detections(frame.copy(), detections), detections

    def draw_detections(self, image: np.ndarray, detections: List[dict]) -> np.ndarray:
        if not detections:
            return image
        color_map = {'bicycle': (255, 0, 0), 'car': (0, 255, 0), 'motorcycle': (0, 0, 255),
                     'bus': (255, 255, 0), 'truck': (255, 0, 255), 'train': (0, 255, 255),
                     'airplane': (128, 0, 128), 'boat': (0, 128, 128)}
        for det in detections:
            x1, y1, x2, y2 = det['bbox'];
            conf = det['confidence'];
            name = det['class_name']
            color = color_map.get(name, (0, 255, 255))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f"{name}: {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(image, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        return image

    def get_model_info(self) -> dict:
        return {"model_path": self.model_path, "device": self.device,
                "conf_threshold": self.conf_threshold,
                "class_count": len(self.class_names),
                "class_names": self.class_names}


# =============================================================================
#  VideoWorker 线程
# =============================================================================
class VideoWorker(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, video_source, yolo_detector):
        super().__init__()
        self.video_source = video_source
        self.yolo_detector = yolo_detector
        self.cap = None
        self.is_running = False
        self.mutex = QMutex()

    def run(self):
        try:
            self.cap = cv2.VideoCapture(self.video_source)
            if not self.cap.isOpened():
                self.error_occurred.emit("无法打开视频源");
                return
            self.is_running = True
            while self.is_running:
                with QMutexLocker(self.mutex):
                    ret, frame = self.cap.read()
                if not ret: break
                if self.yolo_detector.model:
                    output, _ = self.yolo_detector.process_frame(frame)
                else:
                    output = frame
                self.frame_ready.emit(output)
                self.msleep(30)
        except Exception as e:
            logger.error(f"视频线程异常: {e}")
            self.error_occurred.emit(str(e))
        finally:
            if self.cap: self.cap.release()
            self.finished.emit()

    def stop(self):
        with QMutexLocker(self.mutex):
            self.is_running = False


# =============================================================================
#  登录对话框 - 内嵌注册逻辑，点击注册弹出注册界面，不会退出
# =============================================================================
class LoginDialog(QDialog):
    """登录对话框 - 自适应屏幕大小，内置注册弹窗"""

    def __init__(self, user_manager: UserManager, parent=None):
        super().__init__(parent)
        self.user_manager = user_manager
        self.setWindowTitle("系统登录")

        # 获取屏幕尺寸并计算窗口大小（屏幕的40%，最小500x400）
        screen = QDesktopWidget().screenGeometry()
        width = max(int(screen.width() * 0.4), 500)
        height = max(int(screen.height() * 0.5), 450)
        self.setFixedSize(width, height)

        # 居中显示
        self.move((screen.width() - width) // 2, (screen.height() - height) // 2)

        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.init_ui()
        self.apply_style()

    def init_ui(self):
        # 主布局
        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(40, 40, 40, 40)

        # 标题区域
        title_container = QWidget()
        title_layout = QVBoxLayout(title_container)
        title_layout.setSpacing(10)

        # 大标题
        title = QLabel("🔐 基于深度学习的城市道路车型检测系统")
        title.setAlignment(Qt.AlignCenter)
        title_font = QFont("Microsoft YaHei", 18, QFont.Bold)
        title.setFont(title_font)
        title.setStyleSheet("color: #2c3e50; margin-bottom: 5px;")
        title_layout.addWidget(title)

        # 副标题
        subtitle = QLabel("请登录您的账户")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle_font = QFont("Microsoft YaHei", 11)
        subtitle.setFont(subtitle_font)
        subtitle.setStyleSheet("color: #7f8c8d;")
        title_layout.addWidget(subtitle)

        main_layout.addWidget(title_container)

        # 表单区域 - 使用更大的输入框
        form_frame = QFrame()
        form_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 15px;
                padding: 20px;
            }
        """)
        form_layout = QFormLayout(form_frame)
        form_layout.setSpacing(20)
        form_layout.setContentsMargins(30, 30, 30, 30)

        # 设置标签字体
        label_font = QFont("Microsoft YaHei", 11)

        # 用户名输入
        username_label = QLabel("用户名:")
        username_label.setFont(label_font)

        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("请输入用户名")
        self.username_input.setText("admin")
        self.username_input.setMinimumHeight(45)
        self.username_input.setFont(QFont("Microsoft YaHei", 11))
        self.username_input.returnPressed.connect(self.focus_password)

        # 密码输入
        password_label = QLabel("密  码:")
        password_label.setFont(label_font)

        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("请输入密码")
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setMinimumHeight(45)
        self.password_input.setFont(QFont("Microsoft YaHei", 11))
        self.password_input.returnPressed.connect(self.login)

        form_layout.addRow(username_label, self.username_input)
        form_layout.addRow(password_label, self.password_input)

        main_layout.addWidget(form_frame)

        # 按钮区域
        btn_container = QWidget()
        btn_layout = QHBoxLayout(btn_container)
        btn_layout.setSpacing(20)

        self.login_btn = QPushButton("登 录")
        self.login_btn.setDefault(True)
        self.login_btn.clicked.connect(self.login)
        self.login_btn.setMinimumHeight(50)
        self.login_btn.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        self.login_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border-radius: 8px;
            }
            QPushButton:hover { background-color: #2980b9; }
            QPushButton:pressed { background-color: #1a5276; }
        """)

        self.register_btn = QPushButton("注册")
        self.register_btn.clicked.connect(self.open_register)
        self.register_btn.setMinimumHeight(50)
        self.register_btn.setFont(QFont("Microsoft YaHei", 12))
        self.register_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #2ecc71;
                        color: white;
                        border-radius: 8px;
                    }
                    QPushButton:hover { background-color: #27ae60; }
                    QPushButton:pressed { background-color: #1e8449; }
                """)

        btn_layout.addWidget(self.login_btn)
        btn_layout.addWidget(self.register_btn)


        main_layout.addWidget(btn_container)

        # 提示信息区域
        info_frame = QFrame()
        info_frame.setStyleSheet("""
                    QFrame {
                        background-color: #fff3cd;
                        border-radius: 10px;
                        border: 1px solid #ffeaa7;
                    }
                """)
        info_layout = QVBoxLayout(info_frame)
        info_layout.setContentsMargins(20, 15, 20, 15)

        self.info_label = QLabel("💡 默认管理员账户\n用户名: admin\n密码:admin123")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setFont(QFont("Microsoft YaHei", 10))
        self.info_label.setStyleSheet("color: #856404; line-height: 1.6;")
        info_layout.addWidget(self.info_label)

        main_layout.addWidget(info_frame)
        main_layout.addStretch()

        self.setLayout(main_layout)

        # 设置焦点
        self.username_input.setFocus()

    def focus_password(self):
        """用户名输入完成后聚焦到密码输入"""
        self.password_input.setFocus()

    def open_register(self):
        # 内嵌注册弹窗，独立窗口，不会关闭登录界面
        reg_dialog = QDialog(self)
        reg_dialog.setWindowTitle("新用户注册")
        reg_dialog.setFixedSize(480, 420)
        reg_dialog.setWindowFlags(reg_dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        layout = QVBoxLayout(reg_dialog)
        layout.setSpacing(18)
        layout.setContentsMargins(35,35,35,35)

        title = QLabel("📝 用户注册")
        title.setFont(QFont("Microsoft YaHei",16,QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        form = QFormLayout()
        form.setSpacing(15)
        font11 = QFont("Microsoft YaHei",11)

        reg_user = QLineEdit()
        reg_user.setPlaceholderText("用户名，字母/数字/下划线")
        reg_user.setMinimumHeight(42)
        reg_user.setFont(font11)

        reg_pwd = QLineEdit()
        reg_pwd.setPlaceholderText("密码至少6位")
        reg_pwd.setEchoMode(QLineEdit.Password)
        reg_pwd.setMinimumHeight(42)
        reg_pwd.setFont(font11)

        reg_pwd2 = QLineEdit()
        reg_pwd2.setPlaceholderText("重复输入密码")
        reg_pwd2.setEchoMode(QLineEdit.Password)
        reg_pwd2.setMinimumHeight(42)
        reg_pwd2.setFont(font11)

        form.addRow(QLabel("用户名：",font=font11), reg_user)
        form.addRow(QLabel("密　码：",font=font11), reg_pwd)
        form.addRow(QLabel("确认密码：",font=font11), reg_pwd2)
        layout.addLayout(form)

        btn_row = QHBoxLayout()
        btn_submit = QPushButton("完成注册")
        btn_cancel = QPushButton("取消")
        btn_submit.setMinimumHeight(48)
        btn_cancel.setMinimumHeight(48)
        btn_submit.setFont(QFont("Microsoft YaHei",12))
        btn_cancel.setFont(QFont("Microsoft YaHei",12))

        btn_submit.setStyleSheet("""
        QPushButton{background:#27ae60;color:white;border-radius:8px;}
        QPushButton:hover{background:#219653;}
        """)
        btn_cancel.setStyleSheet("""
        QPushButton{background:#95a5a6;color:white;border-radius:8px;}
        QPushButton:hover{background:#7f8c8d;}
        """)

        def submit():
            un = reg_user.text().strip()
            p1 = reg_pwd.text().strip()
            p2 = reg_pwd2.text().strip()
            if not un:
                QMessageBox.warning(reg_dialog,"提示","请填写用户名")
                return
            if len(p1) <6:
                QMessageBox.warning(reg_dialog,"提示","密码长度不能小于6位")
                return
            if p1 != p2:
                QMessageBox.warning(reg_dialog,"提示","两次密码不一致")
                return
            # 调用用户管理类完成注册
            if self.user_manager.add_user(un, p1, False):
                QMessageBox.information(reg_dialog,"成功","注册完成！返回登录页面登录")
                reg_dialog.accept()
            else:
                QMessageBox.critical(reg_dialog,"失败","该用户名已被占用")

        btn_submit.clicked.connect(submit)
        btn_cancel.clicked.connect(reg_dialog.reject)
        btn_row.addWidget(btn_submit)
        btn_row.addWidget(btn_cancel)
        layout.addLayout(btn_row)

        reg_dialog.setLayout(layout)
        reg_dialog.exec_()

    def apply_style(self):
        self.setStyleSheet("""
            QDialog {
                background-color: #ecf0f1;
            }
            QLineEdit {
                padding: 10px;
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                background-color: white;
            }
            QLineEdit:focus {
                border-color: #3498db;
            }
            QLabel {
                color: #2c3e50;
            }
        """)

    def login(self):
        """登录验证"""
        username = self.username_input.text().strip()
        password = self.password_input.text().strip()

        if not username:
            QMessageBox.warning(self, "提示", "请输入用户名")
            self.username_input.setFocus()
            return

        if not password:
            QMessageBox.warning(self, "提示", "请输入密码")
            self.password_input.setFocus()
            return

        if self.user_manager.verify_user(username, password):
            role = "管理员" if self.user_manager.is_admin else "普通用户"
            QMessageBox.information(self, "登录成功", f"欢迎, {username}!\n身份: {role}")
            self.accept()
        else:
            QMessageBox.critical(self, "登录失败", "用户名或密码错误")
            self.password_input.clear()
            self.password_input.setFocus()
# =============================================================================
#  用户管理对话框（仅管理员）
# =============================================================================
class UserManageDialog(QDialog):
    """用户管理对话框"""

    def __init__(self, user_manager: UserManager, parent=None):
        super().__init__(parent)
        self.user_manager = user_manager
        self.setWindowTitle("用户管理")

        # 获取屏幕尺寸
        screen = QDesktopWidget().screenGeometry()
        self.setFixedSize(int(screen.width() * 0.5), int(screen.height() * 0.6))
        self.move((screen.width() - self.width()) // 2, (screen.height() - self.height()) // 2)

        self.init_ui()
        self.refresh_table()

    def init_ui(self):
        layout = QVBoxLayout()

        # 用户表格
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["用户名", "身份", "最后登录", "操作"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setFont(QFont("Microsoft YaHei", 10))
        layout.addWidget(self.table)

        # 添加用户区域
        add_group = QGroupBox("添加新用户")
        add_group.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        add_layout = QGridLayout()

        add_layout.addWidget(QLabel("用户名:"), 0, 0)
        self.new_user_input = QLineEdit()
        self.new_user_input.setMinimumHeight(35)
        add_layout.addWidget(self.new_user_input, 0, 1)

        add_layout.addWidget(QLabel("密码:"), 1, 0)
        self.new_pass_input = QLineEdit()
        self.new_pass_input.setEchoMode(QLineEdit.Password)
        self.new_pass_input.setMinimumHeight(35)
        add_layout.addWidget(self.new_pass_input, 1, 1)

        add_layout.addWidget(QLabel("确认密码:"), 2, 0)
        self.confirm_pass_input = QLineEdit()
        self.confirm_pass_input.setEchoMode(QLineEdit.Password)
        self.confirm_pass_input.setMinimumHeight(35)
        add_layout.addWidget(self.confirm_pass_input, 2, 1)

        self.admin_check = QCheckBox("设为管理员")
        self.admin_check.setFont(QFont("Microsoft YaHei", 10))
        add_layout.addWidget(self.admin_check, 3, 1)

        add_btn = QPushButton("添加用户")
        add_btn.setMinimumHeight(40)
        add_btn.clicked.connect(self.add_user)
        add_layout.addWidget(add_btn, 4, 1)

        add_group.setLayout(add_layout)
        layout.addWidget(add_group)

        # 关闭按钮
        close_btn = QPushButton("关闭")
        close_btn.setMinimumHeight(40)
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

        self.setLayout(layout)

    def refresh_table(self):
        """刷新用户列表"""
        users = self.user_manager.get_user_list()
        self.table.setRowCount(len(users))

        for i, (username, is_admin, last_login) in enumerate(users):
            self.table.setItem(i, 0, QTableWidgetItem(username))

            role_item = QTableWidgetItem("管理员" if is_admin else "普通用户")
            role_item.setForeground(QColor("#e74c3c" if is_admin else "#3498db"))
            self.table.setItem(i, 1, role_item)

            self.table.setItem(i, 2, QTableWidgetItem(last_login))

            # 删除按钮
            if username != "admin":
                del_btn = QPushButton("删除")
                del_btn.setStyleSheet("background-color: #e74c3c; color: white;")
                del_btn.clicked.connect(lambda checked, u=username: self.delete_user(u))
                self.table.setCellWidget(i, 3, del_btn)
            else:
                self.table.setItem(i, 3, QTableWidgetItem("系统账户"))

    def add_user(self):
        """添加新用户"""
        username = self.new_user_input.text().strip()
        password = self.new_pass_input.text()
        confirm = self.confirm_pass_input.text()

        if not username or not password:
            QMessageBox.warning(self, "提示", "请填写完整信息")
            return

        if password != confirm:
            QMessageBox.warning(self, "提示", "两次输入的密码不一致")
            return

        if self.user_manager.add_user(username, password, self.admin_check.isChecked()):
            QMessageBox.information(self, "成功", f"用户 {username} 添加成功")
            self.new_user_input.clear()
            self.new_pass_input.clear()
            self.confirm_pass_input.clear()
            self.admin_check.setChecked(False)
            self.refresh_table()
        else:
            QMessageBox.warning(self, "错误", "用户名已存在")

    def delete_user(self, username: str):
        """删除用户"""
        reply = QMessageBox.question(self, "确认", f"确定要删除用户 {username} 吗？",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            if self.user_manager.delete_user(username):
                QMessageBox.information(self, "成功", f"用户 {username} 已删除")
                self.refresh_table()


# =============================================================================
#  修改密码对话框
# =============================================================================
class ChangePasswordDialog(QDialog):
    """修改密码对话框"""

    def __init__(self, user_manager: UserManager, parent=None):
        super().__init__(parent)
        self.user_manager = user_manager
        self.setWindowTitle("修改密码")

        # 获取屏幕尺寸
        screen = QDesktopWidget().screenGeometry()
        self.setFixedSize(int(screen.width() * 0.3), int(screen.height() * 0.35))
        self.move((screen.width() - self.width()) // 2, (screen.height() - self.height()) // 2)

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(30, 30, 30, 30)

        form_layout = QFormLayout()
        form_layout.setSpacing(15)

        self.old_pass = QLineEdit()
        self.old_pass.setEchoMode(QLineEdit.Password)
        self.old_pass.setMinimumHeight(35)

        self.new_pass = QLineEdit()
        self.new_pass.setEchoMode(QLineEdit.Password)
        self.new_pass.setMinimumHeight(35)

        self.confirm_pass = QLineEdit()
        self.confirm_pass.setEchoMode(QLineEdit.Password)
        self.confirm_pass.setMinimumHeight(35)

        form_layout.addRow("原密码:", self.old_pass)
        form_layout.addRow("新密码:", self.new_pass)
        form_layout.addRow("确认新密码:", self.confirm_pass)

        layout.addLayout(form_layout)

        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("确定")
        ok_btn.setMinimumHeight(40)
        ok_btn.clicked.connect(self.change_password)
        cancel_btn = QPushButton("取消")
        cancel_btn.setMinimumHeight(40)
        cancel_btn.clicked.connect(self.reject)

        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def change_password(self):
        old = self.old_pass.text()
        new = self.new_pass.text()
        confirm = self.confirm_pass.text()

        if not all([old, new, confirm]):
            QMessageBox.warning(self, "提示", "请填写所有字段")
            return

        if new != confirm:
            QMessageBox.warning(self, "提示", "两次输入的新密码不一致")
            return

        # 验证原密码
        if not self.user_manager.verify_user(self.user_manager.current_user, old):
            QMessageBox.critical(self, "错误", "原密码错误")
            return

        if self.user_manager.change_password(self.user_manager.current_user, new):
            QMessageBox.information(self, "成功", "密码修改成功，请重新登录")
            self.accept()
        else:
            QMessageBox.critical(self, "错误", "密码修改失败")


# =============================================================================
#  主界面
# =============================================================================
class MultiVehicleDetectionUI(QMainWindow):
    def __init__(self, user_manager: UserManager):
        super().__init__()
        self.user_manager = user_manager
        self.yolo_detector = YOLODetector()
        self.video_worker = None
        self.current_image = None
        self.is_video_playing = False
        self.folder_path = None
        self.image_list = []
        self.current_idx = -1

        self.init_ui()
        self.init_menu()
        self.setWindowIcon(QIcon())
        self.update_window_title()

        # 窗口最大化显示
        self.showMaximized()

    def update_window_title(self):
        """更新窗口标题显示当前用户"""
        role = "管理员" if self.user_manager.is_admin else "用户"
        self.setWindowTitle(f'基于深度学习的城市道路车型检测系统 - [{self.user_manager.current_user} ({role})]')

    def init_ui(self):
        self.setGeometry(100, 100, 1400, 900)
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        splitter.addWidget(self.create_left_panel())
        splitter.addWidget(self.create_right_panel())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        logger.info("UI 初始化完成")

    def init_menu(self):
        """初始化菜单栏"""
        menubar = self.menuBar()

        # 系统菜单
        system_menu = menubar.addMenu("系统(&S)")

        # 修改密码
        change_pass_action = QAction("修改密码", self)
        change_pass_action.triggered.connect(self.show_change_password)
        system_menu.addAction(change_pass_action)

        system_menu.addSeparator()

        # 退出登录
        logout_action = QAction("退出登录", self)
        logout_action.triggered.connect(self.logout)
        system_menu.addAction(logout_action)

        # 退出系统
        exit_action = QAction("退出系统", self)
        exit_action.triggered.connect(self.close)
        system_menu.addAction(exit_action)

        # 管理菜单（仅管理员可见）
        if self.user_manager.is_admin:
            admin_menu = menubar.addMenu("管理(&M)")

            user_manage_action = QAction("用户管理", self)
            user_manage_action.triggered.connect(self.show_user_manage)
            admin_menu.addAction(user_manage_action)

            # 系统设置
            settings_action = QAction("系统设置", self)
            settings_action.triggered.connect(self.show_settings)
            admin_menu.addAction(settings_action)

        # 帮助菜单
        help_menu = menubar.addMenu("帮助(&H)")
        about_action = QAction("关于", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    # -------------------- 左侧面板 --------------------
    def create_left_panel(self):
        panel = QWidget()
        L = QVBoxLayout(panel)

        # 用户信息显示
        user_group = QGroupBox("当前用户")
        user_group.setStyleSheet("QGroupBox{font-weight: bold; color: #27ae60;}")
        user_layout = QVBoxLayout()

        user_info = QLabel(
            f"用户: {self.user_manager.current_user}\n身份: {'管理员' if self.user_manager.is_admin else '普通用户'}")
        user_info.setStyleSheet("color: #2c3e50; font-weight: bold;")
        user_layout.addWidget(user_info)
        user_group.setLayout(user_layout)
        L.addWidget(user_group)

        # 模型加载
        model_g = QGroupBox("模型加载")
        model_g.setStyleSheet("QGroupBox{font-weight:bold}")
        mv = QVBoxLayout(model_g)
        self.model_path_label = QLabel("模型路径: 未加载")
        self.model_path_label.setWordWrap(True)
        mv.addWidget(self.model_path_label)
        hb = QHBoxLayout()
        self.load_model_btn = QPushButton("加载模型")
        self.load_model_btn.clicked.connect(self.load_model_clicked)
        hb.addWidget(self.load_model_btn)
        self.warmup_btn = QPushButton("模型预热")
        self.warmup_btn.clicked.connect(self.warmup_model)
        hb.addWidget(self.warmup_btn)
        mv.addLayout(hb)
        L.addWidget(model_g)

        # 模型信息
        info_g = QGroupBox("模型信息")
        info_g.setStyleSheet("QGroupBox{font-weight:bold}")
        iv = QVBoxLayout(info_g)
        self.model_info_text = QTextEdit()
        self.model_info_text.setReadOnly(True)
        self.model_info_text.setMaximumHeight(80)
        self.model_info_text.setPlainText("未加载模型")
        iv.addWidget(self.model_info_text)
        L.addWidget(info_g)

        # 图像源
        src_g = QGroupBox("图像源")
        src_g.setStyleSheet("QGroupBox{font-weight:bold}")
        sv = QVBoxLayout(src_g)
        img_line = QHBoxLayout()
        self.image_btn = QPushButton("📷 选择图片")
        self.image_btn.clicked.connect(self.select_image)
        img_line.addWidget(self.image_btn)
        self.folder_btn = QPushButton("📁 选择文件夹")
        self.folder_btn.clicked.connect(self.select_folder)
        img_line.addWidget(self.folder_btn)
        sv.addLayout(img_line)
        self.video_btn = QPushButton("🎬 选择视频")
        self.video_btn.clicked.connect(self.select_video)
        sv.addWidget(self.video_btn)
        self.camera_btn = QPushButton("📹 打开摄像头")
        self.camera_btn.clicked.connect(self.toggle_camera)
        sv.addWidget(self.camera_btn)
        L.addWidget(src_g)

        # 检测参数
        param_g = QGroupBox("检测参数")
        param_g.setStyleSheet("QGroupBox{font-weight:bold}")
        pv = QVBoxLayout(param_g)
        conf_line = QHBoxLayout()
        conf_line.addWidget(QLabel("置信度:"))
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setSingleStep(0.01)
        self.conf_spin.setValue(0.25)
        self.conf_spin.valueChanged.connect(self.update_conf_threshold)
        conf_line.addWidget(self.conf_spin)
        pv.addLayout(conf_line)
        nms_line = QHBoxLayout()
        nms_line.addWidget(QLabel("NMS阈值:"))
        self.nms_spin = QDoubleSpinBox()
        self.nms_spin.setRange(0.1, 0.9)
        self.nms_spin.setSingleStep(0.05)
        self.nms_spin.setValue(0.45)
        nms_line.addWidget(self.nms_spin)
        pv.addLayout(nms_line)
        L.addWidget(param_g)

        # 检测控制
        ctrl_g = QGroupBox("检测控制")
        ctrl_g.setStyleSheet("QGroupBox{font-weight:bold}")
        cv = QVBoxLayout(ctrl_g)
        self.detect_btn = QPushButton("🔍 开始检测")
        self.detect_btn.setEnabled(False)
        self.detect_btn.clicked.connect(self.start_detection)
        cv.addWidget(self.detect_btn)
        nav_line = QHBoxLayout()
        self.prev_btn = QPushButton("⬅ 上一张")
        self.prev_btn.setEnabled(False)
        self.prev_btn.clicked.connect(self.prev_image)
        nav_line.addWidget(self.prev_btn)
        self.next_btn = QPushButton("➡ 下一张")
        self.next_btn.setEnabled(False)
        self.next_btn.clicked.connect(self.next_image)
        nav_line.addWidget(self.next_btn)
        cv.addLayout(nav_line)
        self.stop_btn = QPushButton("⏹️ 停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_detection)
        cv.addWidget(self.stop_btn)
        self.save_btn = QPushButton("💾 保存结果")
        self.save_btn.clicked.connect(self.save_result)
        cv.addWidget(self.save_btn)
        L.addWidget(ctrl_g)

        # 统计信息
        stat_g = QGroupBox("统计信息")
        stat_g.setStyleSheet("QGroupBox{font-weight:bold}")
        stv = QVBoxLayout(stat_g)
        self.fps_label = QLabel("FPS: 0")
        stv.addWidget(self.fps_label)
        self.detection_count_label = QLabel("检测数量: 0")
        stv.addWidget(self.detection_count_label)
        self.video_progress = QProgressBar()
        self.video_progress.setVisible(False)
        stv.addWidget(self.video_progress)
        L.addWidget(stat_g)

        # 日志
        log_g = QGroupBox("检测日志")
        log_g.setStyleSheet("QGroupBox{font-weight:bold}")
        lgv = QVBoxLayout(log_g)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        lgv.addWidget(self.log_text)
        L.addWidget(log_g)
        return panel

    # -------------------- 右侧显示 --------------------
    def create_right_panel(self):
        panel = QWidget()
        rv = QVBoxLayout(panel)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setStyleSheet(
            "QLabel{background-color:#2F3640;border:2px solid #4C78A8;color:#E1E3E6;font-size:18px}")
        self.image_label.setText("加载模型并选择图片或视频开始检测")
        rv.addWidget(self.image_label)
        res_g = QGroupBox("检测结果")
        res_g.setStyleSheet("QGroupBox{font-weight:bold}")
        resv = QVBoxLayout(res_g)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(120)
        self.result_text.setPlainText("暂无检测结果")
        resv.addWidget(self.result_text)
        rv.addWidget(res_g)
        return panel

    # -------------------- 菜单功能 --------------------
    def show_user_manage(self):
        """显示用户管理对话框"""
        dialog = UserManageDialog(self.user_manager, self)
        dialog.exec_()

    def show_change_password(self):
        """显示修改密码对话框"""
        dialog = ChangePasswordDialog(self.user_manager, self)
        if dialog.exec_() == QDialog.Accepted:
            # 密码修改成功后重新登录
            self.logout()

    def show_settings(self):
        """显示系统设置（管理员）"""
        QMessageBox.information(self, "系统设置", "系统设置功能开发中...")

    def show_about(self):
        """显示关于对话框"""
        QMessageBox.about(self, "关于",
                          "基于深度学习的城市道路车型检测系统\n\n"
                          "版本: 2.0\n"
                          "作者: zyf\n"
                          "技术: PyQt5 + YOLOv8 + OpenCV\n\n"
                          )
    def logout(self):
        """退出登录"""
        reply = QMessageBox.question(self, "确认", "确定要退出登录吗？",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.user_manager.logout()
            self.close()
            # 重新显示登录窗口
            show_login_and_main()

    # -------------------- 模型 --------------------
    def load_model_clicked(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 YOLO 模型", "", "YOLO 模型 (*.pt);;所有文件 (*.*)")
        if path:
            if self.yolo_detector.load_model(path):
                self.model_path_label.setText(f"模型路径: {path}")
                self.detect_btn.setEnabled(True)
                self.log_message(f"模型加载成功: {path}")
                info = self.yolo_detector.get_model_info()
                self.model_info_text.setPlainText(
                    f"设备: {info['device']}\n类别数: {info['class_count']}\n置信度阈值: {info['conf_threshold']}")
                QMessageBox.information(self, "成功", "模型加载成功!")
            else:
                QMessageBox.critical(self, "错误", "模型加载失败!")

    def warmup_model(self):
        if self.yolo_detector.model:
            try:
                _ = self.yolo_detector.predict(np.zeros((480, 640, 3), dtype=np.uint8))
                self.log_message("模型预热完成")
                QMessageBox.information(self, "成功", "模型预热完成!")
            except Exception as e:
                self.log_message(f"预热失败: {e}", "error")
                QMessageBox.warning(self, "警告", f"预热失败: {e}")
        else:
            QMessageBox.warning(self, "警告", "请先加载模型!")

    # -------------------- 图像/文件夹 --------------------
    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", "",
                                              "图片文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*.*)")
        if path:
            self.process_image(path)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if folder:
            self.load_folder(folder)

    def load_folder(self, folder):
        exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        self.image_list = []
        for ext in exts:
            self.image_list.extend(Path(folder).glob(ext))
        self.image_list = sorted(self.image_list, key=lambda x: x.name.lower())
        if not self.image_list:
            QMessageBox.warning(self, "提示", "文件夹内未找到图片！")
            return
        self.folder_path = folder
        self.current_idx = 0
        self.prev_btn.setEnabled(True)
        self.next_btn.setEnabled(True)
        self.log_message(f"已加载文件夹：{folder}  （共 {len(self.image_list)} 张）")
        self.process_image(str(self.image_list[0]))
        self.current_image_path = str(self.image_list[0])

    def prev_image(self):
        if not self.image_list:
            return
        self.current_idx = (self.current_idx - 1) % len(self.image_list)
        path = str(self.image_list[self.current_idx])
        self.process_image(path)
        self.current_image_path = path
        self.log_message(f"[{self.current_idx + 1}/{len(self.image_list)}] {Path(path).name}")

    def next_image(self):
        if not self.image_list:
            return
        self.current_idx = (self.current_idx + 1) % len(self.image_list)
        path = str(self.image_list[self.current_idx])
        self.process_image(path)
        self.current_image_path = path
        self.log_message(f"[{self.current_idx + 1}/{len(self.image_list)}] {Path(path).name}")

    def process_image(self, image_path: str):
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("无法读取图像")
            if self.yolo_detector.model is not None:
                output, detections = self.yolo_detector.process_frame(image)
                self.update_statistics(detections)
                self.display_image(output)
                self.current_image = output
                self.update_detection_log(detections, image_path)
                self.log_message(f"图像处理完成: {len(detections)} 个目标检测")
            else:
                QMessageBox.warning(self, "警告", "请先加载模型!")
        except Exception as e:
            self.log_message(f"图像处理失败: {e}", "error")
            QMessageBox.critical(self, "错误", f"图像处理失败: {e}")
        self.current_image_path = image_path

    # -------------------- 视频/摄像头 --------------------
    def select_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择视频", "",
                                              "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*.*)")
        if path:
            self.play_video(path)

    def toggle_camera(self):
        if self.is_video_playing:
            self.stop_detection()
            self.camera_btn.setText("📹 打开摄像头")
        else:
            self.play_camera()
            self.camera_btn.setText("📹 关闭摄像头")

    def play_video(self, video_path: str):
        self.stop_detection()
        self.video_worker = VideoWorker(video_path, self.yolo_detector)
        self.video_worker.frame_ready.connect(self.on_frame_ready)
        self.video_worker.error_occurred.connect(self.on_error_occurred)
        self.video_worker.finished.connect(self.on_video_finished)
        self.video_worker.start()
        self.is_video_playing = True
        self.detect_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.video_progress.setVisible(True)
        self.log_message(f"开始播放视频: {video_path}")
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        self.video_progress.setMaximum(total)

    def play_camera(self):
        self.stop_detection()
        self.video_worker = VideoWorker(0, self.yolo_detector)
        self.video_worker.frame_ready.connect(self.on_frame_ready)
        self.video_worker.error_occurred.connect(self.on_error_occurred)
        self.video_worker.finished.connect(self.on_video_finished)
        self.video_worker.start()
        self.is_video_playing = True
        self.detect_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.video_progress.setVisible(False)
        self.log_message("摄像头已打开")

    def stop_detection(self):
        if self.video_worker:
            self.video_worker.stop()
            self.video_worker.wait()
            self.video_worker = None
        self.is_video_playing = False
        self.detect_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.camera_btn.setText("📹 打开摄像头")
        self.video_progress.setVisible(False)
        self.log_message("检测已停止")

    def on_frame_ready(self, frame: np.ndarray):
        self.display_image(frame)
        self.current_image = frame

    def on_error_occurred(self, msg: str):
        self.log_message(msg, "error")
        QMessageBox.critical(self, "错误", msg)
        self.stop_detection()

    def on_video_finished(self):
        self.stop_detection()
        self.log_message("视频播放完成")
        QMessageBox.information(self, "完成", "视频播放完成!")

    # -------------------- 通用 --------------------
    def display_image(self, image: np.ndarray):
        if image is None:
            return
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)

    def update_statistics(self, detections: List[dict]):
        vehicle_count = len([d for d in detections if d['class_name'] in self.yolo_detector.vehicle_classes])
        self.detection_count_label.setText(f"检测数量: {vehicle_count}")
        if not hasattr(self, 'prev_time'):
            self.prev_time = time.time()
        curr = time.time()
        fps = 1.0 / (curr - self.prev_time) if curr > self.prev_time else 0
        self.prev_time = curr
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def update_detection_log(self, detections: List[dict], src: str):
        if not detections:
            self.result_text.setPlainText("未检测到目标")
            return
        class_count = {}
        for det in detections:
            class_count[det['class_name']] = class_count.get(det['class_name'], 0) + 1
        log_text = f"来源: {src}\n检测时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n总检测数: {len(detections)}\n\n" + "\n".join(
            [f"{k}: {v}" for k, v in class_count.items()])
        self.result_text.setPlainText(log_text.strip())

    def update_conf_threshold(self, v: float):
        self.yolo_detector.conf_threshold = v
        self.log_message(f"置信度阈值更新: {v}")

    def save_result(self):
        if self.current_image is None:
            QMessageBox.information(self, "提示", "没有可保存的结果")
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存结果", "", "图片文件 (*.jpg *.png)")
        if path:
            try:
                cv2.imwrite(path, self.current_image)
                self.log_message(f"结果已保存: {path}")
                QMessageBox.information(self, "成功", f"结果已保存:\n{path}")
            except Exception as e:
                self.log_message(f"保存失败: {e}", "error")
                QMessageBox.critical(self, "错误", f"保存失败: {e}")

    # -------------------- 日志 --------------------
    def log_message(self, msg: str, level: str = "info"):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {msg}")
        if self.log_text.document().lineCount() > 50:
            cursor = self.log_text.textCursor()
            cursor.movePosition(cursor.Start)
            cursor.movePosition(cursor.Down, cursor.KeepAnchor, 5)
            cursor.removeSelectedText()

    # -------------------- 新增：重新检测当前画面 --------------------
    def start_detection(self):
        """重新检测当前画面（图片模式下再用一次当前图）"""
        if self.current_image is not None and hasattr(self, 'current_image_path'):
            self.process_image(self.current_image_path)
        else:
            QMessageBox.information(self, "提示", "请先选择图片或视频")

    def closeEvent(self, event):
        """关闭事件"""
        if self.video_worker:
            self.video_worker.stop()
            self.video_worker.wait()
        event.accept()


# =============================================================================
#  程序入口 - 修复高DPI问题
# =============================================================================
def show_login_and_main():
    """显示登录界面和主界面"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # 关键修复：启用高DPI支持
    # 方法1：Qt5的高DPI属性（兼容性最好）
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    # 方法2：设置环境变量（某些系统需要）
    os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '1'

    # 应用全局样式
    app.setStyleSheet("""
        QMainWindow {
            background-color: #f5f6fa;
        }
        QGroupBox {
            font-weight: bold;
            border: 2px solid #dcdde1;
            border-radius: 8px;
            margin-top: 10px;
            padding-top: 10px;
            background-color: white;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
            color: #2f3640;
        }
        QPushButton {
            background-color: #487eb0;
            color: white;
            border: none;
            padding: 8px;
            border-radius: 4px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #40739e;
        }
        QPushButton:disabled {
            background-color: #7f8fa6;
        }
        QPushButton#danger {
            background-color: #e84118;
        }
        QTextEdit {
            border: 1px solid #dcdde1;
            border-radius: 4px;
            background-color: #f5f6fa;
        }
        QLabel {
            color: #2f3640;
        }
        QMenuBar {
            background-color: #487eb0;
            color: white;
        }
        QMenuBar::item:selected {
            background-color: #40739e;
        }
        QMenu {
            background-color: white;
            border: 1px solid #dcdde1;
        }
        QMenu::item:selected {
            background-color: #487eb0;
            color: white;
        }
    """)

    # 创建用户管理器
    user_manager = UserManager()

    # 显示登录对话框
    login_dialog = LoginDialog(user_manager)

    if login_dialog.exec_() == QDialog.Accepted:
        # 登录成功，显示主窗口
        window = MultiVehicleDetectionUI(user_manager)
        window.show()
        sys.exit(app.exec_())
    else:
        # 用户取消登录，退出程序
        sys.exit(0)


if __name__ == "__main__":
    if not YOLO_AVAILABLE:
        print("警告: YOLO 依赖未安装")
        print("请运行: pip install torch torchvision ultralytics")
        print("或使用 CPU 版本: pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu ")
    print("=" * 60)
    print("多车型检测系统 - Multi Vehicle Detection System")
    print("=" * 60)
    print(
        "功能:\n  • 用户登录与权限管理\n  • 加载 YOLOv5 模型\n  • 单张图片检测\n  • 文件夹翻页浏览\n  • 视频检测\n  • 摄像头实时检测")
    print("=" * 60)
    print("启动登录界面...")
    show_login_and_main()