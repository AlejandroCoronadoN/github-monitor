FROM ubuntu:20.04

# Set the timezone to America/New_York (change to your preferred timezone)
ENV TZ=America/New_York

# Set non-interactive mode for apt
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app/frontend

RUN apt-get update && apt-get install -y nodejs npm && npm install && npm run build
