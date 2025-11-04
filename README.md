# Primer on how to tackle ML problems
A repo with all dependencies and requirements for 9th Nov 2025 session on ML and Kaggle. 

# Table of Contents
* [Installing uv](#installing-uv)
  * [Install](#install)
    * [For macOS and Linux](#for-macos-and-linux)
    * [For Windows](#for-windows)
    * [Docker fanatics](#docker-fanatics)
  * [Post installation steps](#post-installation-steps)
* [Installing packages](#installing-packages)
  * [For uv users](#for-uv-users)
  * [For others](#for-others)
  * [Sanity check](#sanity-check)

# Installing uv
If you have uv installed or a similar package and know how to setup a Python environment (using conda, pyenv, poetry, etc), skip this section and see below where we directly install the packages using your environment of choice.

`uv` is the hot-hot-tip-top Python package manager right now. A package manager in Python is a tool that allows you to manage different dependencies outside of the standard libraries that Python provides.

To focus on machine learning concepts, we'll utilize pre-built Python libraries instead of writing code from scratch (If you want to implement from scratch, kudos to you! But this workshop isn't for that). Some e.gs of popular ML/DL libraries are `scikit-learn`, `pytorch`, etc.

To install `uv`, follow this [link](https://docs.astral.sh/uv/getting-started/installation/). Have distilled the link below for your convenience.

## Install
### For macOS and Linux
Open the `Terminal.app` and paste this command:

If you're a `brew` user, use:
```shell
brew install uv
```

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

If your system doesn't have `curl`, you can use wget:
```shell
wget -qO- https://astral.sh/uv/install.sh | sh
```

### For Windows
Windows comes pre-packaged with a command line utility that takes care of program installation, it's called `winget`.

```shell
winget install --id=astral-sh.uv  -e
```

### Docker fanatics
Yeah! You know who you are! 💀
If you're using Docker, I assume you can read documentation so just follow this [link](https://docs.astral.sh/uv/guides/integration/docker/).

## Post installation steps
Do verify that you can access `uv` in your terminal.

See an example [here](https://i.imgur.com/gOW2CRl.png). If you can see this, please proceed to the next step.

# Installing packages
Before we proceed, please clone this repo at a location of your choice.

```shell
git clone https://github.com/nkapila6/nov9-uaeswe
```

## For uv users
Just navigate to the location where you've cloned the repo via terminal and run the following command.

```shell
uv sync
```

This command will download Python and install all the packages for you.

## For others
The pre-assumption is that you already know how to use your package manager of choice. The repo provides 2 files: `pyproject.toml` from `uv` and a `requirements.txt` file. `uv`s' `pyproject.toml` is compliant with modern PEP standard and should work on package managers that uses a `pyproject.toml` to install packages.

We shall be using Python 3.11 for this workshop. Once you have created the environment in the package manager of your choice, all you need to do is:

```shell
pip install -r requirements.txt
```

If that doesn't work, we only use a few packages for this workshop. You could just run:

```shell
pip install scikit-learn pandas jupyter ipython
```

## Sanity check
To check, have a small Jupyter notebook that checks if everything is fine.

Open `installation-check.ipynb` and run the first cell, you should see an output:
```shell
> Pandas Version: 2.3.3
> Scikit-learn Version: 1.7.2
> NumPy Version: 2.3.4
```

