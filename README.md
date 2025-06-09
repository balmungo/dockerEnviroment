# REQUISITOS

Es necesario contar con [Docker](https://docs.docker.com/) instalado en el equipo y el [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-the-nvidia-container-toolkit) para el acceso a la tarjeta grafica desde docker.

# PREPARACION DEL ENTORNO

Ejecuta el comando en la raiz de tu proyecto o donde tenga el archivo docker-compose.yml, verifica que la carpeta **./src** se encuentre creada.

```bash
sudo docker-compose up
```

# FLUJO DE TRABAJO

Inicia el contenedor con el comando

```bash
sudo docker start py_model
```

verifica que el contenedor este ejecutandose con el comando

```bash
sudo docker ps
```

el status del contenedor debe estar en UP.  
luego ingresa al contenedor con el siguiente comando

```bash
sudo docker exec -it py_model bash
```

se iniciara una consola bash donde podras ejecutar los comando de ejecucion para los arhivos python.  
realiza la prueba con el siguiente comando

```bash
python index.py
```

para salir del contenedor usa el siguiente comando

```bash
exit
```

para detener el contenedor

```bash
sudo docker stop py_model
```

# MODELOS USADOS

[Helsinki-NLP/opus-mt-en-es](https://huggingface.co/Helsinki-NLP/opus-mt-en-es)
[Helsinki-NLP/opus-mt-es-en](https://huggingface.co/Helsinki-NLP/opus-mt-es-en)
[openai/whisper-base.en](https://huggingface.co/openai/whisper-base.en)
[Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
[suno/bark-small](https://huggingface.co/suno/bark-small)
[bertin-project/bertin-roberta-base-spanish](https://huggingface.co/bertin-project/bertin-roberta-base-spanish)
