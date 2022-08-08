# FEniCS-Torch: optimización topológica

## Como ejecutar los ejemplos

La forma recomendada de ejecutar los ejemplos es utilizando un contenerdor Docker, las instrucciones de instalación se pueden consultar en [Get Docker](https://docs.docker.com/get-docker/)

Un vez instalado Docker es posible ejecutarlos haciendo:

`git clone --recurse-submodules https://github.com/hernanponcetta/ps-fenics-torch.git`

`cd ps-fenics-torch`

`docker build -t ps-fenics-torch .`

`docker run -it --name=ps-fenics-torch -v ${PWD}:/home/ps-fenics-torch --rm ps-fenics-torch`

Luego, por ejemplo, para ejecutar wheel_opt.py:

`cd wheel_opt`

`python3 wheel_opt.py`

Para salir y remover el contanedor:

`exit`
