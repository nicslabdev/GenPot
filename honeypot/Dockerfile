FROM python:3.10-slim

WORKDIR /opencanary

# Copy the entire project into the image
COPY . .

# Update pip and install dependencies
RUN pip install --upgrade pip
RUN pip install opencanary

# Overwrite the installed opencanary package with the local version
# (copies the project's opencanary package into the site-packages directory)
RUN cp -r /opencanary/opencanary/* /usr/local/lib/python3.10/site-packages/opencanary/

# Use the official entrypoint
ENTRYPOINT ["opencanaryd"]
CMD ["--dev"]
