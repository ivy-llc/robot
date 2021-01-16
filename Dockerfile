FROM ivydl/ivy:latest

# Install Ivy
RUN git clone https://github.com/ivy-dl/ivy && \
    cd ivy && \
    cat requirements.txt | grep -v "ivy-" | pip3 install --no-cache-dir -r /dev/stdin && \
    cat optional.txt | grep -v "ivy-" | pip3 install --no-cache-dir -r /dev/stdin && \
    python3 setup.py install && \
    cd ..

# Install Ivy Demo Utils
RUN git clone https://github.com/ivy-dl/demo-utils && \
    cd demo-utils && \
    cat requirements.txt | grep -v "ivy-" | pip3 install --no-cache-dir -r /dev/stdin && \
    python3 setup.py install && \
    cd ..

# Install Ivy Mech
RUN git clone https://github.com/ivy-dl/mech && \
    cd mech && \
    cat requirements.txt | grep -v "ivy-" | pip3 install --no-cache-dir -r /dev/stdin && \
    python3 setup.py install && \
    cd ..

# Install Ivy Vision
RUN git clone https://github.com/ivy-dl/vision && \
    cd vision && \
    cat requirements.txt | grep -v "ivy-" | pip3 install --no-cache-dir -r /dev/stdin && \
    python3 setup.py install && \
    cd ..

RUN mkdir ivy_robot
WORKDIR /ivy_robot

COPY requirements.txt /ivy_robot
RUN cat requirements.txt | grep -v "ivy-" | pip3 install --no-cache-dir -r /dev/stdin && \
    rm -rf requirements.txt