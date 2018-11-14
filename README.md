# WACQT - Nanostructure Physics at KTH Royal Institute of Technology

### A "getting started" guide







## Get an account on GitHub
And ask me (Riccardo) to add you as a collaborator to this repository.
Although if you are reading this, you probably have it already.







## Install a **git** client

The simplest way is to install a GUI client. Some options are:
- [GitKraken](https://www.gitkraken.com/): all platforms, really nice interface, the free version has some features locked but from what I can see everything we need is there. You need to login to use it, but you can login with your GitHub account so no need to make a new account. See below for instructions on how to use this.
- [SmartGit](https://www.syntevo.com/smartgit/): all platforms, free for noncommercial use, somewhat less clear interface than GitKraken in my opinion, feel free to try.
- [GitHub Desktop](https://www.gitkraken.com/): Windows and macOS only, I haven't tried it (no Linux), it's the official GitHub client, feel free to try.

On Ubuntu/Debian, you can get around with just the "bare" git client:
```
sudo apt update
sudo apt install git git-gui gitk
```
If you feel like, you could do the same on Windows and macOS: get the "bare" git client on [git's homepage](https://git-scm.com/downloads).







## Clone the repository
### With **GitKraken**
- Start GitKraken. If it is the first time, it will ask you to sign in. Choose *Sign in with GitHub* and follow the instructions in the webpage that will open.
- Choose *File* --> *Clone* --> *GitHub.com*
- Choose a local folder in *Where to clone to*, e.g. `C:\` or `/home/riccardo`
- In *Repository to clone*, choose `rikyborg` --> `wacqt`
- Click *Clone the repo!* at the bottom
- After cloning, a bar will appear at the top saying that the process succeeded. Click on *Open Now*

Done! A new folder is created on your local machine with the content of the repository.

### With the **command line**
if you installed the "bare" git client. For example:
```
cd /home/riccardo
git clone git@github.com:rikyborg/wacqt
cd wacqt
```
For this to work, you might need to configure SSH access to GitHub: [read here](https://help.github.com/articles/connecting-to-github-with-ssh/), or ask me.







## Compile the simulator
### Windows
Install (if you haven't already) [Anaconda](https://www.anaconda.com/download)/[Miniconda](https://conda.io/miniconda.html) for **Python 2.7**. It should work with Pyhton 3.6+ too, but I haven't tried.
Then make sure you have the required packages:
```
conda install numpy scipy matplotlib ipython mingw libpython
```
Open the Anaconda prompt and navigate to the local repository directory, then run `make` to compile the simulator:
```
cd C:\wacqt
make
```
If you got no errors, you're done! Get going with, e.g.:
```
ipython
>>> run test_drive_pulse.py
```



### macOS
Install (if you haven't already) [Anaconda](https://www.anaconda.com/download)/[Miniconda](https://conda.io/miniconda.html) for **Python 2.7**. It should work with Pyhton 3.6+ too, but I haven't tried.
Then make sure you have the required packages:
```
conda install numpy scipy matplotlib ipython
```
~~Download and install from source [SUNDIALS 2.7.0](https://computation.llnl.gov/projects/sundials/download/sundials-2.7.0.tar.gz), follow the instructions in the pdf inside the archive.~~ If the moving surface model works fine for you, you should have SUNDIALS already installed.

Download and install from source the GNU Scientific Library [GSL 2.4](http://ftpmirror.gnu.org/gsl/gsl-2.4.tar.gz):
```
tar -zxvf gsl-2.4.tar.gz
cd gsl-2.4
./configure
make
sudo make install
```
Open the Anaconda prompt and navigate to the local repository directory, then run `make` to compile the simulator:
```
cd /Users/riccardo/wacqt
make
```
If you got no errors, you're done! Get going with, e.g.:
```
ipython
>>> run test_drive_pulse.py
```



### Ubuntu/Debian
Make sure you have the required Python packages.
For Anaconda/Miniconda:
```
conda install numpy scipy matplotlib ipython
```
For system Python:
```
sudo apt install python-numpy python-scipy python-matplotlib ipython
```

Install the development packages for SUNDIALS and the GNU Scientific Library
```
sudo apt install libsundials-dev libgsl-dev
```

Open the terminal and navigate to the local repository directory, then run `make` to compile the simulator:
```
cd /home/riccardo/wacqt
make
```
If you got no errors, you're done! Get going with, e.g.:
```
ipython
>>> run test_drive_pulse.py
```







## Test scripts and file structure
Test/example scripts, showing how to use the simulator and do some analysis with it:
- *test_drive_pulse.py*: drive a 2-tone pulse, and simulate the response over 4 pulse lengths.
- *test_drive_types.py*: show three ways to set up a drive, using a lock-in or by supplying a time-domain array of voltage or its derivative.
- *test_psd.py*: simulate noise in the cavity and qubit, and calculate the power spectral density.
- *test_stiff_nonstiff.py*: compare two solvers for the system, one for stiff equations and one for nonstiff.
- *test_sweep.py*: make a frequency sweep, and compare the amplitude and phase to a linear impedance analysis.

Simulator:
- *simulator.py*: the Python API to the simulator, with lots of docstrings.
- *sim_cvode.c*: the C implementation of the simulator, probably you won't change this at the beginning.

Other:
- *make.bat*: batch script to compile under Windows.
- *Makefile*: compile under Ubuntu/Debian and macOS.
- *win64*: folder with SUNDIALS and GSL headers and precompiled dlls for Windows.







## Finally, more resources about git and GitHub
If you are interested:
- [Git(Hub) cheat sheet](https://services.github.com/on-demand/downloads/github-git-cheat-sheet.pdf)
- [Mercurial for Git Users (and vice versa)](https://www.rath.org/mercurial-for-git-users-and-vice-versa.html)
- [Intro videos on **git**](https://git-scm.com/videos)
- [**git** reference](https://git-scm.com/docs)
- [GitHub help](https://help.github.com/)
- [YouTube tutorials on Git and GitKraken](https://www.youtube.com/channel/UCp06FAzrFalo3txskS1gCfA/playlists)
