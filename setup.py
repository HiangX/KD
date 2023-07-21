from setuptools import setup

setup(
    name='KD',
    version='1.2.1',
    description="'Data Tells Truth'--Survival analysis with knowledge distillation",
    author='Xiu-Shen Wei, He-Yang Xu, ',
    author_email='weixs@njust.edu.cn, xuhy@njust.edu.cn',
    url='https://github.com/HiangX/KDKA',
    setup_requires=[],
    install_requires=['numpy', 'torch', 'scikit-learn', 'tqdm', 'pandas'],
    packages=[
        'KD',
        'KD.datasets',
        'KD.experiment',
        'KD.models',
        'KD.utils',
    ],
    package_dir={
        'KD': 'KD',
        'KD.datasets': 'KD/datasets',
        'KD.experiment': 'KD/experiment',
        'KD.models': 'KD/models',
        'KD.utils': 'KD/utils',
    },
)
