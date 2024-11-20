# Use a lightweight Linux base image
FROM ubuntu:latest

# Set up environment variables to suppress prompts
ENV DEBIAN_FRONTEND=noninteractive

# Update package list and install HMMER
RUN apt-get update && \
    apt-get install -y hmmer && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /data

# Default command to run HMMER help to verify installation
CMD ["hmmsearch", "-h"]
