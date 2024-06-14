import sys
import subprocess
import pkg_resources

def check_installed_libs():
    required = {'tqdm', 'torch', 'torchvision', 'Pillow', 'scikit-learn'}
    installed = {pkg.key for pkg in pkg_resources.working_set}
    print('installed', installed)
    missing = required - installed

    if missing:
        python = sys.executable
        subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)
        print(missing)