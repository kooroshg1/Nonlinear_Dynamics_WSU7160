Installing Python
-----------------------

In order to be able to read and run my notes (versus just display them on the website), you will need to "get some things" installed on your computer.

This is after you get your `github`_ account set up and "join" the class. To do that, create an account on `github`_ and send me your username. I will invite you to the repository. This isn't necessary at the moment, but will be once the class repository becomes private. 

The easiest path to this is to install `Python 3.4`_ via `Anaconda`_. Note that the default is Python 2.7. **You must install* `Python 3.4`_ *for my notes to work fully.**

This proceeds as a normal install on your platform (Mac, Windows, Linux...).

Subsequently you must update some components of Anaconda by using the *conda* command from the *Anaconda Command Prompt*. On Windows, this runs as an actual program.

To ensure conda is as up to date as possible:

.. code-block:: bash
   
   conda update conda
   
Then,

.. code-block:: bash

   conda install jupyter

It will also be useful if you install *Mathjax* via

.. code-block:: bash

  conda install -c https://conda.anaconda.org/anaconda mathjax

This will enable the SymPy notebooks to render nicely.

For convenience, move into the directory with the notes/files. 

.. code-block:: bash

   cd *directory where git put my Notes*

The format for *directory where git put my Notes* varies by platform. You have to figure this out. 

It's a good idea to type ``dir`` of  ``ls`` to see that you are indeed in the directory with all of the files.

This is presuming you have set up your `github`_  account, installed and run the github desktop client (Mac or Windows only), then gone to the Notes (class) repository and clicked on *Clone in Desktop* before all of this.
Cloning is better than copying because it means you can just run the git desktop app to keep your notes up to date with what I have. 


Jupyter is undergoing a massive (favorable) transition right now. To use `Jupyter`_ (the notebook)

.. code-block:: bash

   ipython notebook

It's all in the GUI from here. You just need to play around a bit.

Don't worry, you can't damage my notes. You can always "fix" things by deleting your clone and re-cloning. Play around and have fun!





.. _`github`: http://www.github.com
.. _`Python 3.4`: https://www.python.org/download/releases/3.4.0/
.. _`Anaconda`: http://continuum.io/downloads
.. _`Jupyter`: http://www.jupyter.org
