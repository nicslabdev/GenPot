FROM python:3.10-slim

WORKDIR /opencanary

# Copia tu proyecto completo
COPY . .

# Instala dependencias
RUN pip install --upgrade pip
RUN pip install opencanary

# Sobrescribe toda la instalación de opencanary con tu versión
RUN cp -r /opencanary/opencanary/* /usr/local/lib/python3.10/site-packages/opencanary/

# Usa el entrypoint oficial
ENTRYPOINT ["opencanaryd"]
CMD ["--dev"]
