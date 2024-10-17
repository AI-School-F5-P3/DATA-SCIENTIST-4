# Usa una imagen base de Python
FROM python:3.11-slim

# Establece el directorio de trabajo
WORKDIR /app

# Copia los archivos requeridos (solo requirements.txt inicialmente)
COPY requirements.txt .

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto de la aplicación
COPY . .

# Exponer el puerto que usará la aplicación (por defecto Django usa 8000)
EXPOSE 8000

# Asegura que la carpeta staticfiles esté lista
RUN mkdir -p /app/staticfiles

# Comando para ejecutar la aplicación
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
