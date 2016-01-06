# Nonlinear-Dynamics-and-Vibration
Notes, codes, and other docs for Wright State University ME 7160 Nonlinear Dynamics and Vibration

**Note**: Homework assignments/schedule are still posted on the [WSU class web page][] because of the ability render equations natively. (i.e. I don't have to do file conversions etc.). 

## How to use/get this material
Do **not** click *Download ZIP*.
- It will only get you what is in the repository at this very moment.
- I will be constantly updating, improving, and adding to this repository.
- You will *not* see these updates using that method.

**Do** follow one of these procedures:

1. Good procedure (easier)
  - Install the [GitHub Desktop][] application.
  - Run it, and log in to GitHub from within the application.
  - Go back to the repository for this class
  - Click *Clone in Desktop*
  - On a regular basis:
    - Run the [GitHub Desktop][] application
    - Select *Sync*
  - This will keep your folder in sync with the class notes. 
  

2. Better procedure (harder, but better for communicating with me)
  - On the webpage for the class, in the top right corner of the page, select [Fork][] (follow instructions).
  - You will be able to **update from WSUME7160/Notes/Master** to stay up to date with my edits.
    - *Do this regularly and before you make edits*
  - Using the preceding instructions, clone your fork to your desktop.
  - When you make improvements, save them to your fork and submit a pull request to my repository.
    - For iPython Notebooks (JuPyter notebooks) please select Cell:All Output:Clear then save the notebook
    - Sync with your repository
    - In either your app, or on the website, submit a [pull request][]
  - I *strongly* encourage you to learn how to do this method so that you can be an active participant and learn how *real coding* works.



## File formats used in this respository (let me know if I've missed any- I won't list graphics formats).

| Extension | Description                                                                                                      |
|-----------|------------------------------------------------------------------------------------------------------------------|
| ipynb     | [IPython][] Notebook (Class notes will be in this format)                                                        |
| m         | [Matlab][] script file or function file                                                                          |
| md        | [Markdown][] format                                                                                              |
| pdf       | [Acrobat][] PDF formatted document or IPython Notebook (Class notes will hopefully be converted to this in time) |
| py        | [Python][] Script                                                                                                |
| rst       | [ReStructured Text][] format                                                                                     |

So: Why all of these formats? Any modern engineering class entails programming. Fluency in programming is an essential tool. The best tools are often free- except for the work. Two of the formats (md and rst) are simply simple text files that by following a convention can be easily displayed in beautiful formats, including hyperlinks, etc. Markdown is more of a default on the web, ReST (.rst) is more powerful, though.

Many of you are familiar with Matlab. I've been a Matlab user since 1989. It has a lot to offer, but it is also very constraining as an environment.

Python is where I'm going to. I will be using Python as a language inside IPython notebooks. These notbooks will intertwine notes with code with dynamic illustration in a way you just can't in a PowerPoint demonstration. Embedding my code with my nodes enables me to be more consistent. I highly recommend installing [Anaconda][] (a scientific Python distribution). It's free. That means when you write code, you own it. That's not true of Matlab. Matlab is necessary to run your code, so you are never fully in ownership. Oh, platform indepent, etc. 

I may post PDF files.

If you want to submit to the respository, you can contribute to the growth of the tools we will create. I would like to require this, but we aren't in a lab where I can help like I'd like to. 


[Markdown]: https://help.github.com/articles/markdown-basics
[WSU class web page]: http://cecs.wright.edu/~jslater/classes/nld
[Matlab]: http://www.mathworks.com
[ReStructured Text]: http://docutils.sourceforge.net/docs/ref/rst/restructuredtext.html
[Python]: http://python.org
[Acrobat]: https://acrobat.adobe.com/us/en/acrobat.html
[IPython]: http://ipython.org
[Anaconda]: http://continuum.io/downloads
[GitHub Desktop]: https://desktop.github.com
[Fork]: https://help.github.com/articles/fork-a-repo/
[pull request]: https://help.github.com/articles/be-social/
