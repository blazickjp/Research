---
layout: post
title:  "How to set up IDL Python bridge without admin rights on Linux?"
date:   2019-09-18 15:18:29 +0200
categories: mypost
---

## Introduction

For more than a year [I wrote a post][previous-post] about setting up the [IDL Python Bridge][idl-python-bridge] on Windows without administrative rights. I think it is useful for everyone who is using IDL or Python and have to interact one with another.

Right now I use IDL 8.7.0 (64 bit, Linux) with Python 3.5.2 (32 bit) running on Ubuntu 16.04.6 LTS.

So let's see what to do and how?

## How to do it

### Creating/editing the startup files

As [before on Windows][previous-post], we have to set up variables in two files. Right now it is the [IDL startup file][idl-startup-file] and the .bashrc file, or the startup file of any other shell you use. I use bash, so I write the exact commands for this shell right now, on other shells you have to look for the specific commands yourself.

The location of your [IDL startup file][idl-startup-file] in Linux have to be defined at  your .bashrc file with the following commnand. Assuming your IDL startup filename is ".idl_startup" and is located in your home directory (it can be anything, it's a standard .pro file with the normal IDL syntax) you need to add the following line to your .bashrc.

```bash
export IDL_STARTUP="$HOME/.idl_startup"
```

Since you are already editing your .bashrc file, add the next line to it as well to set up the first pillar of the [IDL Python Bridge][idl-python-bridge]. The line below works for IDL 8.7 and Ubuntu 16.04.6 LTS for sure, I cannot guarantee it for other Linux/IDL versions, but a little search in the file system, and you can locate IDL's installation location.

```bash
export PATH="/usr/local/harris/idl87/bin/bin.linux.x86_64:$PATH"
```

This will contain pythonidlXX.so files - where the XX means your Python major and minor version. E.g. for Python 3.5 it will be 35 -, which are necessary to run IDL from Python.

### Install the idlpy Python module

The next thing to do is to navigate to IDL's lib/bridges library and run the following command.

```bash
cd /usr/local/harris/idl87/lib/bridges
python3 setup.py install --user
```

If the above command fails to execute, try

```bash
python3 setup.py install --home=~/python-packages
```

or

```bash
python3 setup.py install --prefix=~/python-packages
```

In these cases you have to add the folders you give as home or prefix to the **PYTHONPATH** environment variable the same way as you did it with the **IDL_STARTUP** variable.

If all the above installation command fail, you have to ask your system administrator to authorize the following command:

```bash
sudo python3 setup.py install
```

This latter will install *idlpy* for every user. To test whether the installation was successful, type the following snippet into your Python3 command prompt.

```python
>>> from idlpy import *
```

If you got the same message, that you get when you start your IDL session, then the idlpy module has been installed successfully. Otherwise you should get an error message like the following.

```python
ImportError: No module named 'idlpy'
```

If the above import works, you can find more examples for testin the Python to IDL bridge [here][python-to-idl-bridge-webpage].

### Install Python to IDL

In principle, IDL will automatically detect and use the first Python executable, which it finds in your PATH variable. The problem is, that it really looks for an executable named python, and python3 is not good enough for it. So if you want to use Python 3, you have to make a symbolic link, place the location of that link to the beginnning of your path, and you are done. I did it this way.

1. I created a bin directory in my home directory and made a symbolic link to python3 like this (essentially, rename the python3 executable to python).

```bash
mkdir bin && cd bin
ln -s python /usr/bin/python3
```

2. Then I added my bin to my local PATH variable in my .bashrc file.

```bash
export PATH="~/bin:$PATH"
```

To check the installation, you just have to start idl. For the whole test, type the following (I assume you have numpy installed for your Python 3 distribution).

```bash
> idl
IDL> np = Python.Import('numpy')
IDL> a = np.arange(100)
IDL> print,a
```

For this, you should get a vector with 100 elements printed on your screen.

## Final words

Using IDL with Python under linux is not easy, a few hacks need to be applied (especially the python3 --> python renaming), but after that, you have a nice, running bridge in both direction.

I hope you find this tutorial useful. If you have any questions/comments/suggestions, please contact me!

[idl-python-bridge]: http://www.harrisgeospatial.com/docs/Python.html
[idl-startup-file]: http://www.harrisgeospatial.com/docs/StartupFiles.html
[previous-post]: 2018-01-17-how-to-set-up-idl-python-bridge.markdown
[python-to-idl-bridge-webpage]: https://www.harrisgeospatial.com/docs/PythonToIDL.html
