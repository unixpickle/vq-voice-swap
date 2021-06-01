from setuptools import setup

setup(
    name="vq-voice-swap",
    py_modules=["vq_voice_swap"],
    install_requires=["numpy", "torch", "tqdm"],
)
