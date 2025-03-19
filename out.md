### 📌 1. `docker system prune -a`

📖 **Descripción:**
Limpia todo el sistema Docker, eliminando contenedores detenidos, imágenes no utilizadas, volúmenes no referenciados y redes sin uso. El flag `-a` asegura que también se eliminan las imágenes no referenciadas por ningún contenedor.

⚊ **Uso recomendado:** Cuando deseas liberar espacio en disco después de muchas compilaciones y pruebas.

📊 **Ejemplo:**
```bash
docker system prune -a
```
⚠️ **Advertencia:** Este comando eliminará cualquier recurso de Docker que no esté en uso activo. Usa `docker ps` para verificar qué contenedores están corriendo antes de ejecutarlo.

---

### 📌 2. `docker volume prune`

📖 **Descripción:**
Elimina todos los volúmenes no utilizados por al menos un contenedor.

⚊ **Uso recomendado:** Cuando quieres liberar espacio de volúmenes persistentes que ya no necesitas.

📊 **Ejemplo:**
```bash
docker volume prune
```
⚠️ **Advertencia:** Asegúrate de no necesitar los datos almacenados en los volúmenes antes de ejecutar este comando, ya que no es reversible.

---

### 📌 3. `docker build --no-cache`

📖 **Descripción:**
Fuerza la compilación de una imagen Docker sin utilizar la caché intermedia. Es útil cuando sospechas que la caché está causando errores o cuando quieres asegurarte de tener la última versión de las dependencias.

⚊ **Uso recomendado:** Cuando actualizas imágenes o quieres asegurar una compilación limpia.

📊 **Ejemplo:**
```bash
docker build --no-cache -t mi-imagen:latest .
```
⚠️ **Advertencia:** Este proceso es más lento, ya que Docker no reutiliza capas previamente generadas.

---

### 📌 4. `docker exec -it <contenedor> bash`

📖 **Descripción:**
Permite acceder al interior de un contenedor en ejecución mediante una terminal interactiva.

⚊ **Uso recomendado:** Para depurar, inspeccionar archivos de configuración o ejecutar comandos dentro del contenedor.

📊 **Ejemplo:**
```bash
docker exec -it nginx bash
```
⚠️ **Advertencia:** Si el contenedor no tiene bash instalado (por ejemplo, Alpine usa `sh`), ajusta el comando en consecuencia.

---

### 📌 5. `docker inspect <contenedor|imagen>`

📖 **Descripción:**
Proporciona detalles completos en formato JSON sobre un contenedor o una imagen, incluyendo configuraciones de red, volúmenes, puntos de montaje y más.

⚊ **Uso recomendado:** Para auditar configuraciones, rutas de volúmenes, y revisar detalles precisos del entorno.

📊 **Ejemplo:**
```bash
docker inspect nginx
```
⚠️ **Advertencia:** Usa `jq` para formatear la salida si deseas mayor legibilidad.

---

### 📌 6. `docker logs -f <contenedor>`

📖 **Descripción:**
Muestra los logs en tiempo real de un contenedor. El flag `-f` permite seguir los registros a medida que se generan.

⚊ **Uso recomendado:** Para supervisar aplicaciones en ejecución o diagnosticar errores.

📊 **Ejemplo:**
```bash
docker logs -f nginx
```
⚠️ **Advertencia:** Si el contenedor genera demasiados registros, puedes limitar la salida con `--tail <número>`.

---

### 📌 7. `docker cp <contenedor>:<ruta> <destino>`

📖 **Descripción:**
Permite copiar archivos entre un contenedor y el sistema anfitrión.

⚊ **Uso recomendado:** Para extraer archivos de configuración o logs de un contenedor en ejecución.

📊 **Ejemplo:**
```bash
docker cp nginx:/etc/nginx/nginx.conf ./nginx.conf
```
⚠️ **Advertencia:** Asegúrate de que la ruta del contenedor es accesible.

---

### 📌 8. `docker network ls`

📖 **Descripción:**
Muestra todas las redes Docker disponibles, incluyendo sus identificadores, nombres y drivers.

⚊ **Uso recomendado:** Para verificar qué redes están activas y cómo están conectados los contenedores.

📊 **Ejemplo:**
```bash
docker network ls
```
⚠️ **Advertencia:** Si una red no está en uso, puedes eliminarla con `docker network rm <nombre_red>`.

---

### 📌 9. `docker stats`

📖 **Descripción:**
Proporciona métricas en tiempo real del uso de recursos (CPU, memoria, I/O) de los contenedores activos.

⚊ **Uso recomendado:** Para monitorear el rendimiento y detectar cuellos de botella.

📊 **Ejemplo:**
```bash
docker stats
```
⚠️ **Advertencia:** La salida es continua; usa `Ctrl + C` para salir.

---

### 📌 10. `docker export <contenedor> | gzip > backup.tar.gz`

📖 **Descripción:**
Exporta el sistema de archivos de un contenedor a un archivo tar comprimido.

⚊ **Uso recomendado:** Para hacer copias de seguridad del contenido de un contenedor.

📊 **Ejemplo:**
```bash
docker export nginx | gzip > backup-nginx.tar.gz
```
⚠️ **Advertencia:** Esto no guarda los metadatos (variables de entorno, puertos, etc.). Usa `docker commit` si necesitas una imagen completa.

