from setuptools import setup, find_packages

setup(
    name='my_image_loader',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'Pillow',
    ],
    description="A simple image dataloader for PyTorch",
    author="Your Name",
    author_email="your.email@example.com",
)
