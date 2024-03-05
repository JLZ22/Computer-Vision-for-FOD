# Computer Vision for FOD<br>

Leveraging computer vision to aid in foreign object detection in a manufacturing setting.<br>

Currently in development. Limited to no functionality.<br>

This is a research project under Professor Nathaniel Hartman at Purdue University. It is meant to be kept private. DO NOT share with anyone who is not also working with/for Professor Hartman.<br>

## Git Tutorial for Collaborators<br>

This is a repository. A repository is essentially a collection of files and folders. Github stores these repositories on their servers. To get the repository on to your machine, <br> 

- click on the green button that says code and copy the link.<br>
- open the location you want the repository to be stored on you local device in the command line<br>
- type `git clone <the link you copied>`<br>

### Branches<br>

Git is powerful because it allows you to maintain many different sandboxes of your project. Each one of these is called a branch so modifications to one branch will have no effect on others. There are two types of branches, local and remote. A local branch is what is stored on a collaborator's machine while a remote branch is what is stored on Github. You will be editing code on your local branch.<br>

Every repository has a *main* branch which, at least for this project, will be the branch that stores the functioning version of our project. As such, we will never directly edit this branch (more on this in a later section). Instead, create a new branch with a name that most appropriately describes what you are working on (i.e. writing-readme). Branch names should be more or less a general task. After it is complete, the branch can be merged with the main branch (more on this in a later section). This ensures that if anything goes wrong, it does not affect the funcitoning code. Additionally, it prevents multiple contributors from making changes that would impact eachother's code. Below are some important commands to know for branching. 

#### View all the branches of a repository<br>

`git branch -a`<br>

#### Create a new branch<br>

`git branch <insert branch name>`<br>

#### Switch branches<br>

`git switch <insert branch name>`<br>

### Commits<br>

Commits are snapshots of a branch's edit history, and they are created by the developer in order to record and save progress. You typically make a commit after finishing a relatively small subtask.<br>

#### To make a commit, follow the steps below.<br>

- `git add .`<br>
- `git commit -m "<insert brief description of accomplishents>`<br>
- `git push`<br>

#### To commit certain files<br>

- `git add <insert file name 1> <insert file name 2> <...>`<br>

If this does not work, there should be alternative instructions provided in the terminal. 

