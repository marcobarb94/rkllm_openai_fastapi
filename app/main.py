import os
import subprocess
import resource
import sys
import argparse
import uvicorn
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_platform', help='Целевая платформа: например, rk3588/rk3576;')
    parser.add_argument('--rkllm_model_path', help='Абсолютный путь к конвертированной модели rkllm на Linux-устройстве')
    parser.add_argument('--port', help='Абсолютный путь к конвертированной модели rkllm на Linux-устройстве', default=8080)
    args = parser.parse_args()

    if not (args.target_platform in ["rk3588", "rk3576"]):
        print("====== Ошибка: Пожалуйста, укажите правильную целевую платформу: rk3588/rk3576 ======")
        sys.stdout.flush()
        exit()

    if not os.path.exists(args.rkllm_model_path):
        print("====== Ошибка: Пожалуйста, укажите точный путь к модели rkllm, учтите, что это должен быть абсолютный путь на устройстве ======")
        sys.stdout.flush()
        exit()

    # Настройка фиксированной частоты
    command = "sudo bash fix_freq_{}.sh".format(args.target_platform)
    subprocess.run(command, shell=True)

    # Установка ограничения на количество файловых дескрипторов
    resource.setrlimit(resource.RLIMIT_NOFILE, (102400, 102400))

    # Инициализация модели RKLLM
    print("=========инициализация....===========")
    sys.stdout.flush()
    target_platform = args.target_platform
    model_path = args.rkllm_model_path
    rkllm_model = RKLLM(model_path, target_platform)
    print("Инициализация RKLLM успешно завершена!")
    print("==============================")
    sys.stdout.flush()

    # Запуск приложения Flask
    app.run(host='0.0.0.0', port=int(args.port), threaded=True, debug=False)
    uvicorn.run(app, port=10000)

    print("====================")
    print("Вывод модели RKLLM завершен, освобождение ресурсов модели RKLLM...")
    rkllm_model.release()
    print("====================")