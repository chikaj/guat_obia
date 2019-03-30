FROM continuumio/miniconda3

# Set the ENTRYPOINT to use bash
# (this is also where you’d set SHELL,
# if your version of docker supports this)
ENTRYPOINT [ “/bin/bash”, “-c” ]

EXPOSE 5050

RUN apt-get update && apt-get install -y \
 gdal-bin libgdal-dev \
&& rm -rf /var/lib/apt/lists/*

# Use the environment.yml to create the conda environment.
ADD environment.yml /tmp/environment.yml
WORKDIR /tmp
RUN conda env create -f /tmp/environment.yml

ADD . /code

# Use bash to source our new environment for setting up
# private dependencies—note that /bin/bash is called in
# exec mode directly
WORKDIR /code
RUN [ “/bin/bash”, “-c”, “source activate geo3” ]

WORKDIR /code
RUN [ “/bin/bash”, “-c”, “source activate geo3” ]

# We set ENTRYPOINT, so while we still use exec mode, we don’t
# explicitly call /bin/bash
CMD [ “source activate geo3” ]
