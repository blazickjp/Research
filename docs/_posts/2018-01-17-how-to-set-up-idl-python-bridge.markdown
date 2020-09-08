---
layout: post
title:  "How to set up IDL Python bridge without admin rights on Windows?"
date:   2018-01-17 17:41:08 +0100
categories: mypost
---
# Introduction

My problem is for days now, that IDL's built-in XML handler classes are terrible. Just terrible. They are either totally low-level written, or totally uncomfortable, or just need a normal example code, but in the end, for me they are useless. Fortunately IDL has a feature sinc IDL 8.5 called [IDL Python Bridge][idl-python-bridge], with which you can connect Python with IDL both ways. So you can use IDL from Python, or Python from IDL. This feature was rather premature in IDL 8.5, however, in IDL 8.6 it seems a lot better. But you have to set a few environmental variables to make it work, which is not possible for me, because I am using an office machine, and I do not have admin rights. It's a 64-bit Windows 7 by the way.

So this tutorial is about how to connect IDL 8.6.1 (64-bit, Windows) with WinPython 3.6.1.0Qt5 (64-bit) without admin rights!

Ready! Set! Go!!!

# How to do it?
Basically, you only have to set up variables in two files. One is the env.bat file in the WinPython environment, and the other is the IDL startup file in the IDL environment (I assume, that you have properly installed both of them). Let's see what to put into them!

Add this to your env.bat file in the C:\Program Files\Winpython-64bit-3.6.1.0Qt5\scripts\.

    set PATH=%PATH%;c:\Program Files\Harris\IDL86\bin\bin.x86_64

And add this to your [IDL startup file][idl-startup-file] probably in your HOME directory (or anywhere, where you put the IDL startup file, see the link).

    SETENV,'PYTHONHOME=C:\ProgramFiles\WinPython-64bit-3.6.1.0Qt5\python-3.6.1.amd64'
    SETENV,'PATH='+GETENV("PATH")+'C:\Program Files\WinPython-64bit-3.6.1.0Qt5\python-3.6.1.amd64\Scripts;C:\Program Files\WinPython-64bit-3.6.1.0Qt5\python-3.6.1.amd64;C:\Program Files\WinPython-64bit-3.6.1.0Qt5;C:\Program Files\WinPython-64bit-3.6.1.0Qt5\scripts;C:\Program Files\Harris\IDL86\lib\bridges;'

And in the end, run the following command from the command line:

    > cd C:\Program Files\Harris\IDL86\lib\bridges
    > python setup.py install

And you are done! Now you can try your shiny new bridge from both ends.

Python:

```python
>>> from idlpy import *
>>> import numpy.random as ran
>>> arr = ran.rand(100)
>>> p = IDL.plot(arr, title='My Plot')
>>> p.color = 'red'
>>> p.save('myplot.pdf')
>>> p.close()
```

IDL:
```idl
IDL> ran = Python.Import('numpy.random')
IDL> arr = ran.rand(100)  ; call "rand" method
IDL> plt = Python.Import('matplotlib.pyplot')
IDL> p = plt.plot(arr)    ; call "plot", pass an array
IDL> void = plt.show(block=0)  ; pass keyword
```

That is all for today! I hope you found it useful! If you have any questions, write me! :)

[idl-python-bridge]: http://www.harrisgeospatial.com/docs/Python.html
[idl-startup-file]: http://www.harrisgeospatial.com/docs/StartupFiles.html
