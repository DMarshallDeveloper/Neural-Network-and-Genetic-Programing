This folder contains my code for part 2 of the assignment, the Genetic Programming.
To run my code, after extracting the zip file, simply follow these steps:

- Open a new console and copy and paste (ctrl-SHIFT-v) the following commands into it
- cd 'Downloads/COMP307 Assignment 2 Final/Part 2 - Genetic Programming/'
- pip install pygraphviz
- python a2Part2.py

NOTE: if pygraphviz does not install for any reason, there is a second python file that 
runs the algorithm without outputting the tree, to call this do the following steps:
- Open a new console and copy and paste (ctrl-SHIFT-v) the following commands into it
- cd 'Downloads/COMP307 Assignment 2 Final/Part 2 - Genetic Programming/'
- python a2Part2NoGraph.py

The code should run as expected and generate the output which is stored in sampleoutput.txt
Note that the tree.pdf is also part of the output, but will change each time you run the program

Libraries used:
- operator
- random
- pandas
- numpy
- deap
- pygraphviz
To install these libraries if they are absent (shouldn't be needed except for possibly pygraphviz)
- Use the pip install command
e.g. pip install pygraphviz

