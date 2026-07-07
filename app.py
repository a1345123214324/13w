import os
import subprocess
import sys

import scraper_weibo
import scraper_bilibili
import scraper_reddit
import data_cleaning
import data_processing
import setting.concat_csv


def auto_check_and_install_dependencies():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    requirements_path = os.path.join(current_dir, "..", "requirements.txt")

    if not os.path.exists(requirements_path):
        print("未找到 requirements.txt，跳过自动检测。")
        return

    # 读取 requirements.txt，解析出所有需要安装的库名
    required_packages = []
    with open(requirements_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # 过滤掉空行、注释或者本地路径(file:///)
            if line and not line.startswith("#") and "@" not in line:
                # 提取库的名字（比如从 torch==2.3.0 中切出 torch）
                package_name = line.split("==")[0].split(">=")[0].split("<=")[0].strip()
                if package_name:
                    required_packages.append(package_name)

    # 挨个检查这些库在当前环境里是否存在
    missing_packages = []
    for package in required_packages:
        # 特殊转换：有些库安装名和导入名不一样，这里做个兼容
        import_name = package
        if package.lower() == "scikit-learn":
            import_name = "sklearn"
        elif package.lower() == "opencv-python":
            import_name = "cv2"

        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package)

    # 发现有漏网之鱼，立刻启动精准补全
    if missing_packages:
        print(f"检测到当前环境缺少以下 {len(missing_packages)} 个必要依赖: {missing_packages}")
        print("正在为您自动补全，请耐心等待...\n")

        base_cmd = [sys.executable, "-m", "pip", "install"]

        try:
            # 如果缺少的库里包含 torch，单独拉出来装 GPU 加速版
            if "torch" in missing_packages:
                print("正在下载兼容 GPU 加速的 PyTorch...")
                torch_cmd = base_cmd + ["torch", "torchvision", "torchaudio", "--index-url",
                                        "https://download.pytorch.org/whl/cu121"]
                subprocess.run(torch_cmd, check=True)
                # 装完 torch 后，从缺失列表里把它踢掉
                missing_packages = [p for p in missing_packages if
                                    p != "torch" and p != "torchvision" and p != "torchaudio"]

            # 如果除了 torch 还有其他好兄弟没装，打包一起装了
            if missing_packages:
                print(f"正在安装剩余缺失的库: {missing_packages}...")
                # 也可以直接针对缺失的库挨个安装
                subprocess.run(base_cmd + missing_packages, check=True)

            print("\n所有缺失依赖已全部精准补齐！正在开启主程序...\n")

        except subprocess.CalledProcessError as e:
            print(f"自动环境部署失败，请检查网络！错误信息: {e}")
            sys.exit(1)
    else:
        print("[环境检查] 完美！当前环境里的库和 requirements.txt 完全吻合，秒开！")



def step_scraper():
    print("爬取数据中：")
    scraper_bilibili.scraper_bilibili()
    print("B站爬取完成")
    scraper_weibo.scraper_weibo()
    print("微博爬取完成")
    scraper_reddit.scraper_reddit()
    print("reddit爬取完成")
    print("数据爬取完成")

def step_cleaning():
    print("清理数据中：")
    data_cleaning.data_cleaning()

def step_processing():
    print("开始执行BERT模型批量预测：")
    data_processing.predict_comments()

def step_concat():
    print("开始合并CSV数据")
    setting.concat_csv.concat_csv()

# 直接运行即可，不需要配环境
def step_dashboard():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dashboard = os.path.join(current_dir, "web_dashboard.py")
    # 使用 subprocess.run，这样不会杀死当前的 Python 进程，更安全可控
    # 显式指定用当前虚拟环境里的 python 去跑 streamlit
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard, "--server.port", "8501"])
    except KeyboardInterrupt:
        print("\nWeb 看板服务已停止。")


def main():
    print("================ PyWarVoices 项目管理工具 ================")
    print("1. 运行完整管道 (爬虫 + 清洗 + 推理 + 合并 + 启动看板)")
    print("2. 直接启动 Web 看板 (使用已有数据)")
    print("========================================================")

    choice = input("请输入选项 (1 或 2): ").strip()

    if choice == "1":
        auto_check_and_install_dependencies()
        step_scraper()
        step_cleaning()
        step_processing()
        step_concat()
        step_dashboard()
    elif choice == "2":
        auto_check_and_install_dependencies()
        step_dashboard()
    else:
        print("输入错误，程序退出。")
if __name__ == "__main__":
    main()