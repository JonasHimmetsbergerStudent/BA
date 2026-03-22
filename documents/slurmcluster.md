General information
The dataLAB cluster is a shared resource that enables a wide range of research projects. Because many users depend on the same hardware and software environment, it is essential to establish a common understanding of how to use the system responsibly. This course introduces the basic etiquette of working on the dataLAB cluster. Its purpose is to ensure that all users can access resources efficiently and fairly. By following a few simple conventions, you will not only avoid common pitfalls but also contribute to a smoother experience for your colleagues.

The course is not intended as a comprehensive manual but as a structured introduction. It provides the essentials while pointing you to further resources. After explaining how to access the cluster, including the creation of an SSH key pair, it covers Python environments and then moves on to job submissions with Slurm. You will learn the relevant terms, how to run and monitor calculations, and how to use /scratch directories for temporary storage. The "Cluster etiquette" chapter outlines how limited resources can be shared productively, even as the user base grows. "Slurm advanced" introduces strategies for handling larger workloads and useful shortcuts such as bash aliases. The course condludes with a short overview of Apptainer, a container management system for installing software and ensuring reproducibility, followed by a collection of all useful links.

Completion of this course is mandatory for gaining or keeping access to the dataLAB cluster. It is fully online and can be completed at your own pace. Experienced HPC users should find the material straightforward but will still benefit from learning about dataLAB-specific conventions. New users, in turn, will gain the tools and knowledge needed to work productively on the cluster without disrupting others. Each chapter ends with either a short quiz or a confirmation that you have reviewed the material. You have unlimited attempts to achieve the required 80 % score where applicable.

Accessing the cluster

Getting access
The whole process is documented on the colab page, including screenshots. For that reason, this will be kept brief.

If you follow these steps and your access is approved by the admins, you are good to go:

Log into the dataLAB to appear in the system.
Generate an ssh key pair using
ssh-keygen -t ed25519 -C user@tuwien.ac.at
use the email address that appears on your dataLAB profile and choose a fitting name for the key file, e.g., ~/.ssh/id_datalab, where ~ points to your home directory on a Linux machine.

Add the public part of your ssh key ~/.ssh/id_datalab_pub to your dataLAB profile. It might take a couple of minutes for the key to be propagated and work.
Join the matrix channel.
Write a post in the channel there with your name, role (e.g., PhD student), email, and supervisor. If you are part of a workgroup with a collective share also add this to the post.
Hint: Access for bachelor and project students needs to be requested by the supervisor to be granted an exception.

Connecting to the cluster
You can either access the cluster via ssh or start a JupyterHUB session.

ssh
For authentication, you need to have the private part of the ssh key pair you set up above. Best practice is to create an entry in your ~/.ssh/config on Linux or associate the key to the ssh connection on Windows, for example, by using putty.

Host datalab
    Hostname cluster.datalab.tuwien.ac.at
    User USERNAME
    AddKeysToAgent yes
    IdentityFile ~/.ssh/id_datalab
With this entry present you can connect to the cluster with

ssh datalab
Alternatively, you can forego the config and pass all information when connecting

ssh -i ~/.ssh/id_datalab user@cluster.datalab.tuwien.ac.at
JupyterHUB
Hint: This creates an interactive session (see Slurm chapter) and, therefore, should not be left to idle to not block resources unnecessarily.

Use the link https://jupyter.datalab.tuwien.ac.at/ to start a session. You can log in to the page using TU Wien single-sign-on (SSO)

This image shows the drop-down menu that becomes available once logged in to the dataLAB JupyterHUB.
The drop-down menu contains the available hardware options.

content of the image: 
"Server Options

Select a job profile:

Head Node - No GPUs

GPU-a40 - 1 GPU - 2 CPUs - 64G Memory - 48 Hours

GPU-l40s - 1 GPU - 2 CPUs - 64G Memory - 48 Hours

GPU-a100 - 1 GPU - 2 CPUs - 64G Memory - 48 Hours

GPU-a100s - 1 GPU - 2 CPUs - 64G Memory - 48 Hours

GPU-h100 - 1 GPU - 2 CPUs - 64G Memory - 48 Hours"

If your chosen hardware is available, a session will start once the needed resources are available. For guidance on how to use Jupyter notebooks check out the available documentation.

Hint: Jupyter sessions count against the maximum number of concurrent GPUs available per user.

Useful links
dataLAB colab
dataLAB login
ssh key generation (GitHub docs)
putty ssh client
Jupyter notebooks (Project Jupyter)
Icon Abstimmung
Acitivity completion - Cluster access
I confirm that I have read and thought through the Access to dataLAB section and understand the conditions for using the cluster.


Python environments

General
On the dataLAB cluster, Python is not preinstalled. That way conflicts between the many different versions of Python needed for various software packages are avoided. Instead of relying on a single system Python, every user is expected to create their own environments. An environment is a self-contained installation of Python with the packages you need for your work.

To create such environments we recommend micromamba, a lightweight tool that is fully compatible with the larger conda ecosystem, but faster and easier to use. You can, of course, instead use any alternative that you are already familiar with, e.g., miniconda or uv.

Example setup and usage
The commands used in the following are taken from the micromamba installation guide.

Installation
Micromamba can be installed directly using

"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
Hint: While not strictly necessary for this small package, it is sensible to perform installations and builds on a compute node, without requesting a GPU. See the next chapter on how to do this.

You will be prompted for installation details (e.g., paths) and can simply go with the defaults.

At the end of this, an initialization block will be added to your shell setup, e.g., .bashrc if you are using bash. After reloading the shell or reconnecting micromamba will be usable.

Hint: If you use multiple package managers or load-heavy initialization, you can move that block to a different file (e.g., .init_mamba, note the "." following naming conventions). When needed, initialize on the command line or in a shell script via

source .init_mamba
Since your home directory is accessible from all compute nodes, the same holds for your environments. This means you only have to install your package manager of choice and set up needed environments once for the cluster and not for each node.

Environments
Create the environment myenv with Python in version 3.12

micromamba create -n myenv python=3.12
Activate the new environment

micromamba activate myenv
To install Python package requirements I recommend using pip pypi. You can of course use the micromamba command instead. Note that either approach installs compatible versions of packages into the current environment.

pip install numpy pandas scipy
is equivalent (and preferable) to

micromamba install -n myenv numpy pandas scipy
Hint: I find the micromamba command tediously long to type, so I usually define an alias in my .bashrc. alias mm="micromamba" allows you to use mm in place of micromamba for all commands.

Environment setups can be exported to yml files, e.g., to replicate on different machines.

micromamba env export > myenv.yml
This includes a section for the pip-installed packages, which will be downloaded and built on recreation.

micromamba create -n myenv_copy -f myenv.yml
To delete an environment use

micromamba env remove -n myenv
 
Housekeeping
Package managers such as micromamba or pip store a lot of temporary data. To avoid using up your home-data quota on that, it is sensible to periodically clean those data.

Useful commands for this are

micromamba clean --all --yes  # empty the micromamba cache
pip cache purge  # empty the pip cache
Important notes
Do not auto-load specific environments at login (e.g., in your `.bashrc`). This will not only slow down your own login, but also put avoidable load on the login node, potentially hindering other users. Worst case (and this has happened a couple of times) this might block that login node for all other users (and yourself).
Individual environments can get rather large, depending on what packages are installed. Remember to remove environments (or packages) you no longer need.
Periodically remove temporary files / empty the caches (micromamba and pip). See Housekeeping above.
Useful links
micromamba (a tiny version of the mamba package manager)
miniconda (a free, miniature installation of Anaconda Distribution)
uv (An extremely fast Python package and project manager, written in Rust.)
venv (Python built-in virtual environments)
Icon Test
Quiz - Python Environments
You have an unlimited amount of attempts to achieve 80 % of the available points. All chapter quizzes need to be passed in order to complete the course.

Slurm basics

General
This section is not meant to regurgitate the details of Slurm, HPC hardware, and similar, since many exhaustive resources are already available (see useful links below). Instead, the goal is to enable you to submit standard calculations, to ask the right questions of search engines or LLMs of your choice, and, hopefully, critically look at what might be LLM hallucinations.

To ensure efficient and fair usage of cluster resources, schedulers are used. On the dataLAB cluster the scheduler is Slurm (formerly an acronym, but this is no longer the case), which offers exhaustive documentation, user guides, tutorials, and cheat sheets. Some of these are linked below (see useful links). This chapter introduces the basics, enabling you to get an overview, submit calculations, and get more use out of the available online resources.

By using this scheduler, the maximum number of resources and runtime per user is limited to better enable a fair distribution of resources. Also, by submitting calculations as batch jobs, the computational resources are only blocked for as long as they are needed.

Before submitting jobs it is practical to get a sense of the available resources. The dataLAB colab lists the relevant information.

Basic terms
To get the relevant information out of the documentation, tutorials, and this page, some terms need to be introduced.

Job
A job is a unit of work you submit to the scheduler. These typically are batch scripts that run for hours or days, but also interactive sessions. Each job has a unique job ID.

Queue / Partition
A partition is a collection of nodes with similar hardware. In the dataLAB partitions are specific to the GPU type. Queue is often used to mean partition, since submitted jobs are queued to be executed on specific partitions. You choose a partition when submitting a job. The various partitions give access to different hardware, which is reflected in the available CPUs and memory.

Node
A node is a physical or virtual machine that is part of the cluster. It provides memory, CPUs, and/or GPUs. Slurm decides which node runs a job, but specific nodes can be excluded by the user. We will encounter an example of this in "Slurm advanced".

Resources
These are the things the user requests from the scheduler, e.g., number of CPUs, number of GPUs, amount of memory, runtime etc. A specific partition or node type might also be called a resource.

Bookkeeping
Slurm provides several commands that allow you to keep an overview of your pending, active or past jobs and also give you information about available resources.

sinfo
gives a summary of the cluster status. What partitions are available? How is the utilization and status of the nodes (idle / allocated / down)? Allows you to check for partitions that allow your job to run immediately.

squeue
lists all currently submitted jobs. These may be in various states, including "R" for running and "PD" for pending. Among the many options available for this command, `-u $USER` limits the list to your jobs.

sacct
shows accounting information for jobs that have already finished or are running. Allows you to quickly check if your jobs succeeded or failed.

scontrol
can be used to gather various information. scontrol show job <jobid> is very useful to see more details about a specific job, e.g., memory used.

There are of course many more commands and a large number of additional options. Please refer to the Slurm documentation for that.

Running calculations
Once you know how to check available resources and monitor jobs, the next step is to actually run calculations. This can be done in two ways: by submitting batch jobs or by starting an interactive session. Batch jobs are the standard approach, while interactive sessions are useful for testing and debugging.

Submitting a batch job
Batch jobs are submitted to Slurm as scripts using the sbatch command. A script specifies which resources are needed and what commands should be executed.

As discussed above, the various partitions on the dataLAB cluster provide different hardware. Always make sure not to unnecessarily reserve more memory or CPUs than actually needed. If all CPUs are taken, idle GPUs cannot be utilized, since every job needs at least one CPU.

Parameters
These (and many others) can be either written into the submit script as in the example below or passed to sbatch in the command line or a mix of both. Their usage is demonstrated in examples below.

--nodes
The number of nodes to be used for the job (max 2, typically 1).
--ntasks
The CPU resources (cores) needed. This coincides with the MPI ranks, if MPI is used (eg, mpirun).
--cpus-per-task
The CPU cores required for each task (multithreading). The default is 1. If increased, the total number of CPU resources used is ntasks times cpus-per-task.
--gpus
Request GPUs (up to 8, typically 1 is sufficient).
--test-only
Checks your submit script for inconsistencies, performs no computation.
The following script runs without a GPU, which can, for example, be useful for performing data backup.

#!/bin/bash

#SBATCH --partition=GPU-a40
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --time=01:00:00  # request 1 h of max runtime
#SBATCH --mem=30 GB  # memory size required per node (system memory, not GPU memory)


# always include this, may provide useful information to the admins
echo $SLURM_JOB_NODELIST

# your calculation
echo "Finished running a CPU-only job."
To submit this job, first test the script via

sbatch --test-only myjob.sh
If everything checks out, submit it with

sbatch myjob.sh
Starting an interactive session
Use interactive sessions sparingly and, most importantly, do not block resources by letting them idle. Such jobs (interactive sessions are jobs for the Slurm manager) can be started using srun. If the requested resources are available, the session will start immediately, otherwise the request will be queued and shows up in the squeue output.

srun --partition=GPU-a40 --nodes=1 --ntasks=1 -J rsync --pty /bin/bash
Using notebooks on JupyterHUB can be understood as interactive sessions. Details on this follow in a later section.

Example scripts
Run Python script on GPU and 4 CPU cores. No MPI parallelization:
#!/bin/bash

#SBATCH -J python
#SBATCH --partition=GPU-a40  # select a partition
#SBATCH --nodes=1  # run on 1 node
#SBATCH --ntasks=1  # perform 1 task
#SBATCH --cpus-per-task=4  # use 4 threads
#SBATCH --gpus=a40:1  # 1 GPU

# Log where the calculation is executed
echo $SLURM_JOB_NODELIST  # identify the node
nvidia-smi -L  # identify the card(s)

# initialize your package manager of choice
source ~/.init_mamba

# activate the Python environment
micromamba activate myenv

# perform the calculation (4 cores, 1 GPU)
python super_cool_gpu.py
Run Python script with `mpirun`, using 2 GPUs and 4 CPU cores per MPI rank:
#!/bin/bash

#SBATCH -J python
#SBATCH --partition=GPU-a40  # select a partition
#SBATCH --nodes=1  # run on 1 node
#SBATCH --ntasks-per-node=2  # perform 2 tasks (= 2 MPI ranks)
#SBATCH --cpus-per-task=2  # 2 threads for each rank
#SBATCH --gpus=a40:2  # 2 GPUs

# Log where the calculation is executed
echo $SLURM_JOB_NODELIST  # identify the node
nvidia-smi -L  # identify the card(s)

# initialize your package manager of choice
source ~/.init_mamba

# activate the Python environment
micromamba activate myenv

# perform the calculation (4 cores, 2 GPUs)
mpirun -n 2 python super_cool_mpi.py
Important notes
Avoid interactive jobs. They have their use cases, for example, debugging or quick prototyping, but should never be used for production runs.
Avoid blocking resources. You can tell Slurm the number of GPUs, CPUs, and memory to reserve for your calculation. Unused GPUs are unavailable if another job unnecessarily takes all available CPUs.
Rules typically need to become more restrictive when more users are involved or if the existing users do not adhere to fair sharing.
Useful links
Slurm documentation
Slurm cheat sheet (two-pager of important commands)
Slurm quick start guide (also from official documentation)
Compact reference (USC, center for advanced research computing)
ASC batch jobs (ASC, Austrian scientific computing specific intro)
Slurm tutorial (University of Innsbruck)
Icon Test
Quiz - Slurm basics
You have an unlimited amount of attempts to achieve 80 % of the available points. All chapter quizzes need to be passed in order to complete the course.

Storage and data transfer

General
Before introducing the storage setup of the dataLAB cluster, an important note: None of the cluster filesystems is meant for long-time storage!

Disk access (IO) can put unnecessary strain on the cluster. Therefore, please take note of these simple rules:

Use a scratch directory for IO-heavy calculations.
Transfer large data (e.g., backups) on compute nodes. Use a less-powerful partition and request one CPU, either as a batch job or interactively.
Avoid data redundancy. If multiple people in your group use the same datasets, grant each other access or use a share directory.
Types of storage
/home, quota of 100 GiB
Typically used to store code repositories and environments. Also holds the pip and micromamba cache. Clean this often to not run into the quota.
/share fair-use
Each user can create their own directory there. It is sensible for groups to collect all their user shares in a group directory. This simplifies access management.
/scratch temporary storage on compute nodes
Except for virtual nodes (hostnames starting with "i-"), all compute nodes provide temporary storage. It is good practice, to have calculations write there during runtime. At the end of your job (or periodically for long runs), copy results back to /share or another persistent location. Details on how to use scratch directories are given later in this chapter.
Data accessibility
/home and /share can be accessed from the login node and any compute node. Each compute node can access its own /scratch directory and, additionally, all /scratch dirs can be accessed from the login node. Data in /scratch is not guaranteed to be kept for longer than 24 hours after job completion. Unless you do this yourself at the end of your job script, the data will not be lost immediately though.

Filesystem	Size / Quota	Lifetime	Used for
/home	100 GiB	persistent	Code, environments, etc.
/share	Fair-use	persistent	Data, shared resources
/scratch	Varies	temporary	IO at runtime
Scratch usage
Currently, the creation of job-specific /scratch directories is not automated. For that reason, the user has to take care of this themselves.

Hint: While not strictly enforced, utilizing job-specific /scratch directories is strongly recommended.

[...]  # header (sbatch parameters

# create a job-specific scratch dir if it does not already exist
scratch_dir="/scratch/${USER}_${SLURM_JOB_ID}"
if [ ! -d "${scratch_dir}" ]; then
    mkdir ${scratch_dir}
    chmod 700 ${scratch_dir}
fi

cd ${scratch_dir}
export results_dir=/share/${USER}/data

[...]  # calculation (write to scratch_dir)

# copy relevant data from scratch_dir to results_dir
cp -r ${scratch_dir}/* ${results_dir}

# if you created very large amounts of data and the copying worked, delete the temp directory
# be careful, data cannot be restored
rm -r ${scratch_dir}
Data transfer
As already mentioned above, make sure to use a compute node for data transfer to avoid load on the login node. Either using an interactive session or, if a lot of data will be moved over a longer time, by submitting a batch job.

Hint: You do not need to request a GPU for that and typicall one CPU core suffices.

The most convenient and efficient method is rsync, which will only copy data that is newer or not present at the target location.

rsync -avP -e ssh source user@remote:/target/
The above command

preserves permissions, timestamps, symbolic links, etc.  (parameter -a)
prints information about the ongoing process (-v)
shows the progress (-P)
uses ssh to connect to the target (-e ssh), you might need to add a specific port
copies the directory source and its content, source/ would copy only the content
copies into the directory /target on the remote machine
You can also rsync in the opposite direction, i.e., from a remote machine to local.

An alternative is scp. This works like the cp command but uses ssh to connect to a remote machine.

scp -r source user@remote:/target/
This recursively copies source and its content (-r) to the remote target.

Useful links
rsync documentation (man page)
scp documentation (man page)
Icon Test
Quiz - Storage and data transfer

Cluster etiquette

The low entry barrier and fast availability of resources on the dataLAB cluster should not be taken for granted. At the moment, the user base is relatively small, so jobs typically start quickly and resources appear plentiful. This is not the usual situation on a large HPC systems, where queues can be long and competition for resources is high.

Good etiquette ensures that everyone benefits from the cluster fairly, both now and as the user base grows. Following the rules of fair use, respecting quotas, and being considerate with resource help maintain a smooth experience for all users and reduce the support workload for the administrators.

What is fair use?
Fair use means utilizing the cluster in a way that respects both other users and the infrastructure. It is not defined by strict numerical limits but by good judgement and shared responsibility. If everyone respects fair use, strict rules can be kept to a minimum, allowing for more flexibility.

Request only the resources you actually need.
Submit jobs efficiently (arrays instead of flooding the scheduler).
Use the right type(*) of GPU.
Use the right filesystem.
Monitor your jobs and clean up unused data.
(*) Most calculations do not require or even utilize the fastest GPUs with the most memory.

Dos and don'ts
Dos
Do submit your calculations to the scheduler, even short test runs.
Do monitor your jobs and cancel ineffective / erronuous ones.
Do make use of the /scratch directories.
Don'ts
Don't run calculations or large file transfers on the login nodes.
Don't request excessive resources "just in case" (this includes CPUs and memory).
Don't treat the cluster as long-term backup storage.
Rules
/home quota of 100 GB.
Users are locked out of submitting calculations.
Maximum of 8 GPUs per user concurrently.
Enforced by the scheduler.
Limited runtime for interactive jobs.
Currently monitored, will most likely have to be enforced. Multi-day interactive sessions block resources without using them for approximately 12 hours a day.
Icon Abstimmung
Activity completion - Cluster etiquette
I confirm that I have read and thought through the Cluster etiquette section, understand the conditions for using the cluster, and will stay within the limits of fair use.

Slurm advanced

General
Slurm offers considerably more features than are introduced in the chapter "Slurm basics" or can be touched upon here. Here, we present selected functionality which not only provides you with more control over your calculations, but also facilitates the fair-use approach. Additionally, you can find some convenient bash aliases for common Slurm commands towards the end.

Advanced job management
Beyond the basics of submitting and monitoring jobs, Slurm also provides mechanisms to handle more complex workflows. Two particularly useful features are job arrays and job dependencies. They allow you to launch many similar jobs efficiently and to control the order in which jobs are executed, without manual intervention. Together, these tools help streamline chained calculations and automated pipelines.

Job arrays
In addition to submitting a large number of jobs in a clean way, job arrays also enable you to restrict your own resource usage. If, for example, 100 instances of an evaluation need to be executed, but you do not want to block all resources availabe to you concurrently, you can limit how many array tasks run at the same time.

Hint: Remember that sbatch parameters can be set in the command line or in the submit script or both.

To submit 100 instances of a job with at most 5 running concurrently, use

sbatch --array=1-100%5 myjob.sh
It makes no difference, which of the 5 active jobs finishes first, the next will take its place as soon as possible. Each job instance has an ID that can be accessed at runtime via

$SLURM_ARRAY_TASK_ID
and, for example, used as the random seed (or basis for calculating a random seed) for training ML model committee members.

How many jobs will be running in reality also depends on the available resources at any given time.

Hint: You might want or need to cancel all the jobs of an array that are still pending, while letting the active jobs finish. The following command does exactly that for JobID 12345.

squeue --array --user=$USER --states=PENDING --noheader --format="%i" | grep '^12345' | xargs -r scancel
Job dependencies
In many workflows, post-processing or evaluation needs to be done after, e.g., training a model. If the post-processing should be attempted naturally is contingent on whether the training was successfully completed. Instead of periodically checking / reacting to email notifications, all required jobs can be submitted at the same time using job dependencies for coordination. For other cases, you may want to start alternative calculations if previous attempts fail.

The most commonly used dependency types are

afterany: start after a job finishes (no matter the outcome)
afterok: start only if the job succeeds
afternotok: start only if the job fails
The simplest (and impractical) approach to dependencies is knowing a JobID (1234) and submitting the subsequent job (part2.sh) as

sbatch --dependency=afterok:1234 part2.sh
Scripted dependencies
Typically, it might be convenient to submit a series of jobs within a single script (see the "Example scripts" below).

Example scripts
Sequence of dependencies (each submit script can itself contain an array)
In this example, first an array of model trainings is submitted and the Slurm JobID read from the return value. Using this JobID, the rest of the workflow is submitted in sequence with each only starting once (and if) the previous step ends successfully. The for-loop calls the same script each time, but passes in a different parameter.

#!/usr/bin/env bash

# submit training (can be a job array)
JOBID=$(sbatch submit_train.sh | awk '{print $4}')

# sequentially run evaluations, each only if the previous step succeeded.
for param 'convolve' 'simulate' 'plot'
do
    JOBID=$(sbatch --dependency=afterok:$JOBID submit_eval.sh $param | awk '{print $4}')
done
Array job with random seed
#!/bin/bash
#SBATCH -J calc_array
#SBATCH --partition=GPU-a40
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:a40:1
#SBATCH --time=12:00:00
#SBATCH --array=1-20%5

# max. 5 instances at the same time
# each instance runs for 12 hours at the longest

# random seed is the product of Job_ID and Task_ID
export SEED=$((SLURM_ARRAY_TASK_ID * SLURM_JOB_ID))

# Python script call
python run_calculation.py --seed=${SEED}
Practical aliases
A collection of commands that have turned out to be useful or convenient. You can add these to your ~/.bashrc file which is loaded at login. If you are already connected, the ~/.bashrc can be executed again using

source ~/.bashrc
only list your own jobs
alias qstat="squeue -u $USER"
start a minimal interactive session
alias run_interactive="srun -J interact -n 1 --pty /bin/bash"
a more exhaustive `squeue` call
alias sqlong="squeue -o '%.16i %.9P %.24j %.8u %.2t %.10M %.6D %C %N'"
show detailed information about used and available CPU and GPU resources
alias sqos="sinfo -O 'PartitionName:12,StateCompact:8,NodeAI:12,Memory:12,CPUs:8,CPUsState,Reason:14,Gres:25,NodeList,GresUsed'"
Useful links
Slurm job array support (official documentation)
sbatch command (official documentation)
Slurm dependencies (NIH Biowulf)
Icon Test
Quiz - Slurm advanced

Apptainer basics

General
The dataLAB cluster runs apptainer (formerly Singularity), which allows you to utilize containers. This is not only useful for installing software (e.g., a version of CUDA not available on the cluster) but also to better ensure reproducibility of your calculations and results. Furthermore, containers can more easily be shared to other clusters or with your colleagues on the same cluster.

This chapter is not meant to give an in-depth guide, but aims to show the overall workflow as an entrypoint to existing documentation.

Hint: Build your Apptainer containers in an interactive session on a compute node to avoid unnecessary load on the login node.

Container management
An exhaustive guide to Apptainer containers was made available by the University of Wisconsin-Madison, including this section on HPC systems.

The apptainer command is available to all users. To see the installed version call from the command line interface (CLI)

apptainer version
Definition file
The build process is non-interactive, which means that the definition file needs to be set up to work without user intervention. Typically, you build on top of an existing container image and add what is required.

For example, a very base Rockylinux 9 image enriched by essential (and not so essential) tools.

BootStrap: docker
From: rockylinux:9

%post
    echo "Enabling EPEL repository..."
    dnf install -y epel-release | tee /var/log/epel-install.log

    echo "Updating system and installing packages..."
    dnf update -y | tee /var/log/dnf-update.log
    dnf install -y \
        less \
        vim \
        wget \
        make \
        gcc \
        tar \
        bzip2 \
        unzip \
        bc \
        | tee /var/log/package-install.log

    echo "Cleaning up..."
    dnf clean all

    echo "Verifying installation..."
    which gcc | tee /var/log/which-gcc.log
    which bc | tee /var/log/which-bc.log
    rpm -q gcc | tee /var/log/rpm-gcc.log
    rpm -q ncdu | tee /var/log/rpm-ncdu.log

    echo "Build process finished."

%environment
    export PATH="/opt/tools:$PATH"
    export LC_ALL="C.UTF-8"
    export LANG="C.UTF-8"

%labels
    Author "RW"
    Version "0.1"
    Description "Rocky Linux 9 container with essential tools"

%help
    Usage: apptainer exec rocky.sif bc <<< '2^10'
Looking at the packages that are installed, e.g., less, indicates how basic the underyling bootstrap image is. Nevertheless, this example illustrates how an apptainer definition file is structured.  Detailed explanations on the different parts are available and will not be reiterated here. Many more examples are available online and can be found, for example, via the useful links.

Build
The containers are built using the CLI, resulting in .sif binaries if successful. For the above script, saved as rocky.def, the command is

apptainer build rocky.sif rocky.def
Usage
Containers can be used in different ways. This includes calling one command from within the container but also allows to work from within it.

When submitted jobs include Apptainer usage, this typically means that a specific version of a software is called. The syntax for this call is

apptainer exec rocky.sif bc <<< '2^10'
which executes a quick calculation in the bash calculator bc.

To interactively test a container you can launch a shell that works within:

apptainer shell -e rocky.sif
Useful links
Apptainer(official page)
User guide (University of Wisconsin-Madison)
Support for Docker (official page)
Icon Abstimmung
Activity completion - Apptainer
I confirm that I have taken note of the Apptainer basics chapter and am aware that Apptainer is the prefered method for utilizing software not installed on the cluster.
Collected links

Please make full use of this course, the provided resources, and linked guides. They should always be your first stop for support and will save you time. The administrator will appreciate this, as it leaves them more available to help with issues that are not already covered.

dataLAB colab
dataLAB login
ssh key generation (GitHub docs)
putty ssh client
Jupyter notebooks (Project Jupyter)
micromamba (a tiny version of the mamba package manager)
miniconda (a free, miniature installation of Anaconda Distribution)
uv (An extremely fast Python package and project manager, written in Rust.)
venv (Python built-in virtual environments)
rsync documentation (man page)
scp documentation (man page)
Slurm documentation
Slurm cheat sheet (two-pager of important commands)
Slurm quick start guide (also from official documentation)
Compact reference (USC, center for advanced research computing)
ASC batch jobs (ASC, Austrian scientific computing specific intro)
Slurm tutorial (University of Innsbruck)
Slurm job array support (official documentation)
sbatch command (official documentation)
Slurm dependencies (NIH Biowulf)
Apptainer(official page)
User guide (University of Wisconsin-Madison)
Support for Docker (official page)

Erstellt von Simon Brandstetter, zuletzt aktualisiert am 22.10.2025  Lesedauer: 6 Minute(n)
Create snapshot
Welcome to the GPU Cluster introduction page of the TU-Wien Datalab. 
This page will provide you with a short overview about the cluster, login methods and a short introduction to Slurm. Further information about Slurm is available on the official documentation.

You can join our Matrix channel: https://matrix.to/#/#gpu:tuwien.ac.at

Access to the Cluster
To gain access to our Slurm Cluster complete this Tuwel course, to learn the basics of how to work with our cluster and Slurm in general:

dataLAB Cluster Essentials
password: dataLABW2025



Most of the following information is included in the Tuwel course.

Table of Contents
Access to the Cluster
Table of Contents
User Creation
Login to the Cluster
SSH (Secure Shell):
Jupyter Notebook:
Slurm Cluster
Cluster Nodes:
Storage:
Partitions (queues):
Default parameters:
Using GPUs in your job:
Submitting a batch job:
Interactive session
Job submission
Basic Slurm commands
Links
FAQ:
Q: 
A:
Q: 
A:
User Creation
To access the Slurm cluster you first need to have completed the above mentioned Tuwel course and have a datalab user account in https://login.datalab.tuwien.ac.at where you can configure your ssh public key. You can create this account yourself if you have TU Wien credentials, otherwise you need to contact us.

Once you created your account you need to write us on the Matrix channel or create a Jira request to provide you with the needed access and permissions to login.

If your an employee of the TU Wien please send your TISS Business Card for confirmation. It should look something like this: https://tiss.tuwien.ac.at/person/xxxxxx.html
If you are a student writing on your master thesis or PHD, please give us a confirmation of that, in the best case by your Supervisor.
Bachelor Students are currently not granted access to the cluster, due to the limited resources. Exceptions can be requested for by a supervisor. 






Login to the Cluster
SSH (Secure Shell):
Afterwards you may remotely login to the head node of the cluster where you put your apply your jobs



$ ssh cluster.datalab.tuwien.ac.at -l username
Jupyter Notebook:
If you want to access the cluster via a Jupyter notebook you can instead call https://jupyter.datalab.tuwien.ac.at from  your browser you need to have a password set in the previous step when creating your user.

Slurm Cluster
Cluster Nodes:
Hostname	CPU Type	CPUs	Cores/CPU	Threads	GPU Type	Bus	Count	GPU Mem	IB
a-a100-o-1	AMD	2	32	2	a100	SXM4	8	80GB	YES
a-a100-o-2	AMD	2	64	2	a100	SXM4	8	80GB	YES
a-a100-os-3	AMD	2	64	2	a100s	SXM4	8	40GB	YES
a-a100-os-4	AMD	2	64	2	a100s	SXM4	8	40GB	YES
a-a100-q-5	AMD	2	64	2	a100	SXM4	4	80GB	YES
a-a100-q-6	AMD	2	32	2	a100	SXM4	4	80GB	YES
a-a100-qs-7	AMD	2	64	2	a100s	SXM4	4	40GB	YES
a-a100-qs-8	AMD	2	64	2	a100s	SXM4	4	40GB	YES
a-a40-o-1	
AMD

2	24	2	a40	PCIe 4	8	48GB	NO
a-l40s-o-1	
AMD

2	32	2	L40S	PCIe 5	8	48GB	NO
a-l40s-o-2	
AMD

2	32	2	L40S	PCIe 5	8	48GB	NO
DGXs	 
dgx-h100-1	Intel	2	56	2	h100	SXM5	8	80GB	YES
dgx-h100-2	Intel	2	56	2	h100	SXM5	8	80GB	YES
dgx-h100-3	Intel	2	56	2	h100	SXM5	8	80GB	YES
VMs

 

ivm-a40-q-2	Intel	2	10	2	a40	PCIe 4	4	48GB	NO
ivm-a40-q-3	Intel	2	10	2	a40	PCIe 4	4	
48GB

NO

avm-a40-o-4	AMD	2	40	1	a40	PCIe 4	8	48GB	NO
avm-v100-d-5	AMD	2	8	1	v100	PCIe 4	2	32GB	NO
avm-a100-qs-9	AMD	2	34	1	a100s	PCIe 4	4	40GB	NO
avm-a100-d-10	AMD	2	18	1	a100	PCIe 4	2	80GB	NO
Storage:
The main Storage used is Ceph as a network file system that includes /home and /share on all servers

/home for user's home directory
Limited to 250GB
/share to have groups directory for collaborations (please reach us for special groups and permissions)
No limit. Please try to avoid duplicate datasets and instead put them in some of the public folder like "/share/models".
/scratch is a local nvme storage RAID 0 it exist only on GPU nodes - they are mounted on head nodes as NFS as well.
The scratch storage is the fastest option. Please put any files created during a job there and move them to /home or /share afterwards. Copying datasets to the scratch before execution can speed things up for longer jobs.  
Partitions (queues):
GPU-v100: i-v100-o-1,i-v100-h-2,i-v100-q-[3-4]
GPU-a40: a-a40-o-1
GPU-a100: a-a100-o-[1-2],a-a100-q-[5-6]
GPU-a100s: a-a100-h-9,a-a100-os-[3-4],a-a100-qs-[7-8]
Default parameters:
There are default parameters set it is different to each partition for example:

Max Nodes per job are 2.
Max job time is 7 days.
Default time for any job is 24 hours, to overwrite this value you should set in your script the time limit option --time which can be maximum of 168 hours.
Default MEM per CPU, default CPU per GPU, default memory per GPU, these can vary between partitions but you can overwrite them in your script.
To facilitate creating your Slurm bash script parameters, you may use this code generator platform: https://code-gen.datalab.tuwien.ac.at/ and also check the command sbatch man pages for more information.

Using GPUs in your job:
to use GPU you should request it in your script as in the following example:



#SBATCH --gres=gpu:a100s:2  # recommended way - to choose the right GPU in case of a server with multiple GPU types that can be mentioned in multiple partitions
or
#SBATCH --gres=gpu:2
when specifying the type of the GPU (which is recommended) it should correspond with the partition provided as in the following:



#SBATCH --partition=GPU-a100s
#SBATCH --gres=gpu:a100s:2
or
#SBATCH --partition=GPU-a6000
#SBATCH --gres=gpu:a6000:2
Submitting a batch job:
Processing computational tasks requires submitting jobs to the Slurm scheduler. Slurm offers two commands to submit jobs: sbatch and srun.

Always use sbatch to submit jobs to the scheduler, unless you need an interactive terminal. Otherwise only use srun within sbatch for submitting job steps within an sbatch script context. The command sbatch accepts script files as input.

Scripts should be written in bash, and should include the appropriate Slurm directives at the top of the script telling the scheduler the requested resources. Read on to learn more about how to use Slurm effectively.

Interactive session
For interactive session as in the following example:



$ srun --pty bash
You might have more options to the srun command please check the command man pages.

Job submission
You simply create you bash script and run it with the command sbatch how in the following examples:

A small example job submission script called hello-world.sh 



#!/bin/bash

#SBATCH --partition=GPU-a100 		# select a partition i.e. "GPU-a100"
#SBATCH --nodes=2 					# select number of nodes
#SBATCH --ntasks-per-node=8 		# select number of tasks per node
#SBATCH --time=00:15:00 			# request 15 min of wall clock time
#SBATCH --mem=2GB 					# memory size required per node

echo "hello, world"
the part between the shebang and your script by putting #SBATCH [option] you can define the resources needed for your script and this is tunable please check the sbatch command man pages for other options.

to submit the job:



$ sbatch hello-world.sh
Basic Slurm commands
sinfo gives information which partitions (queue) are available for job submission. 
scontrol is used to view SLURM configuration including: job, job step, node, partition, reservation, and overall system configuration. Without a command entered on the execute line, scontrol operates in an interactive mode and prompt for input. With a command entered on the execute line, scontrol executes that command and terminates.
scontrol show job 32 shows information on the job with number 32.
scontrol show partition shows information on available partitions.
squeue to see the current list of submitted jobs, their state and resource allocation. Here is a description of the most important job reason codes returned by the squeue command.
scancel 32 to cancel your job number 32.
Links
The official Slurm documentation: https://slurm.schedmd.com/documentation.html
The Slurm documentation of the Vienna Scientific Cluster: https://wiki.vsc.ac.at/doku.php?id=doku:slurm






FAQ:
Q: 
I am scheduling an interactive session via srun --pty bash but I don't see the GPUs

A:
Scheduling interactive sessions in like scheduling any job in slurm in order to have your interactive session with GPU you need to specify additional parameters for the partition and the GPU resources for example: 
srun -p GPU-a100 --gres=gpu:a100:2 -w a-a100-o-1 --pty bash  here you are scheduling on the partition GPU-a100 and you are asking to have 2 GPUs of type a100 and optionally you want that to be on the node a-a100-o-1

Q: 
I'm missing a certain software on the cluster

A:
Have you tried using Aptainer? This allows you to create a custom container with all the software you need. 
Here is a guide you can follow written by the University of Wisconsin–Madison: https://chtc.cs.wisc.edu/uw-research-computing/apptainer-hpc