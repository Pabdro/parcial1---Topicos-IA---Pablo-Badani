# Parcial 1 Topicos en IA - Reconocimiento de curso
### Pablo Alejandro Badani Zambrana - 55789

# Reconocimiento de curso - Descripcion: 
La aplicacion presente de reconocimiento facial se especializa en el control de un curso para evitar personas que no sean parte del curso y tener documentada la entrada de cada estudiante.

# Reconocimiento de curso - Funcionalidad:
Para poder utilizarla se debe hacer correr el archivo app.py, despues se tiene que ir al link proporcionado para irnos a la interfaz de fastAPI donde tendremos que poner la ruta "/docs" para poder ingresar a los diferentes endpoint:
- /status .- Para ver informacion general de la aplicacion y modelo.
- /annotate .- Para subir la imagen del alumno a reconocer.
- /faces .- Para mostrar diferentes caracteristicas de los alumnos como la deteccion de ojos, boca y nariz.
- /reports .- Nos crea un registro .csv de los alumnos que fueron detectados donde nos mostrará el nombre respectivo del que se le sacó la foto, fecha y hora, entre otros datos, cabe destacar que solo se puede usar despues de haber utilizado el /annotate 

# Reconocimiento de curso - Problemas:
No se pudo fusionar los endpoints /annotate y /faces para una deteccion mas completa por lo que estan aparte para evitar bugs y por ultimo el modelo entrenado tiene problemas al detectar a Dylan confundiendolo con Gabriel posiblemente por falta de datos.