nlp
===

NLP research code for writing a paper on dependancy parsing.

Python Setup
------------

Setup Virtualenv if you havn't already

	easy_install pip
	pip install virtualenv virtualenvwrapper

Add the following to your `.bashrc`file

	export WORKON_HOME=$HOME/.virtualenvs
	export PROJECT_HOME=$HOME/dev
	source /usr/local/bin/virtualenvwrapper.sh

then run it `source ~/.bashrc` to 

create a new environment for the NLP project

	mkproject -r requirements.txt nlp
	workon nlp