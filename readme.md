FEniCS-Torch: Topological Optimization
How to Run the Examples
The recommended way to run the examples is by using a Docker container. Installation instructions can be found at Get Docker.

Once Docker is installed, you can run the examples by doing the following:

git clone --recurse-submodules https://github.com/hernanponcetta/ps-fenics-torch.git

cd ps-fenics-torch

docker build -t ps-fenics-torch .

docker run -it --name=ps-fenics-torch -v ${PWD}:/home/ps-fenics-torch --rm ps-fenics-torch

Then, for example, to run wheel_opt.py:

cd wheel_opt

python3 wheel_opt.py

To exit and remove the container:

exit
