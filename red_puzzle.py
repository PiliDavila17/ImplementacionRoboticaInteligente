import rclpy 
from rclpy.node import Node 
import cv2 
import numpy as np 
from cv_bridge import CvBridge 
from sensor_msgs.msg import Image 
from std_msgs.msg import String  # Importar el mensaje de String
from ultralytics import YOLO  # Importar la librería YOLO

# Definición de la clase Vision, que hereda de Node
class Vision(Node): 

    def __init__(self, usar_camara_robot): 
        # Inicializa la clase base con el nombre del nodo
        super().__init__('vision_semaforo_node') 
        self.bridge = CvBridge()  # Inicializa el puente para convertir entre ROS y OpenCV
        self.usar_camara_robot = usar_camara_robot
        self.model = YOLO("/home/victorn65/ros_ws/red_neuronal/best.pt")  # Carga el modelo de YOLO desde el archivo

        #### SUSCRIPTORES ####
        if self.usar_camara_robot:
            # Suscripción al tópico de imagen de la cámara del robot
            self.sub = self.create_subscription(Image, "video_source/raw", self.camera_callback, 10)
        else:
            # Captura de video desde la cámara local
            self.cap = cv2.VideoCapture(0)

        #### PUBLICADORES ####
        self.image_received_flag = False  # Bandera para verificar si se ha recibido una imagen
        self.signal_pub = self.create_publisher(String, 'segnal', 10)  # Publicador para las señales

        ### TEMPORIZADOR ###
        dt = 0.1  # Intervalo de tiempo del temporizador
        self.timer = self.create_timer(dt, self.timer_callback)  # Crear el temporizador
        self.current_senial = ""  # Variable para almacenar la última señal publicada
        self.last_detection_time = None  # Tiempo de la última detección
        self.senial = None  # Variable para la señal detectada

    # Callback para la suscripción a la cámara
    def camera_callback(self, msg): 
        try:  
            # Convertir el mensaje de imagen a una imagen OpenCV
            self.cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")  
            self.image_received_flag = True  # Bandera para indicar que se recibió una imagen
        except: 
            self.get_logger().info('Failed to get an image')  # Registro de error en caso de fallo al recibir la imagen

    # Callback del temporizador
    def timer_callback(self): 
        # Verificar si se usa la cámara del robot o la cámara local
        if self.usar_camara_robot:
            if self.image_received_flag: 
                image = self.cv_img.copy()  # Copiar la imagen recibida
            else:
                return
        else:
            ret, image = self.cap.read()  # Leer la imagen de la cámara local
            if not ret:
                self.get_logger().info('Failed to get an image')  # Registro de error en caso de fallo al recibir la imagen
                return
        
        # Procesar la imagen con YOLO
        results = self.model.predict(image, imgsz=640, conf=0.4)

        # Extraer información de los resultados y dibujar las cajas delimitadoras
        for result in results[0].boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0])  # Coordenadas de la caja delimitadora
            ancho = x2 - x1  # Ancho de la caja delimitadora
            alto = y2 - y1  # Alto de la caja delimitadora
            confianza = result.conf[0]  # Confianza de la detección
            clase = result.cls[0]  # Clase de la detección
            nombre_clase = self.model.names[int(clase)]  # Nombre de la clase detectada

            # Dibujar la caja delimitadora y la etiqueta en la imagen
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{nombre_clase} {confianza:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(image, f"{ancho}", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Verificar el nombre de la clase y el tamaño de la caja para determinar la señal
            if nombre_clase == "stop" and ancho >= 24:
                self.senial = "stop"
            elif nombre_clase == "turn_left" and ancho >= 30:
                self.senial = "turn_left"
            elif nombre_clase == "turn_right" and ancho >= 27:
                self.senial = "turn_right"
            elif nombre_clase == "give_way" and ancho >= 30:
                self.senial = "give_way"
            elif nombre_clase == "work_in_progress" and ancho >= 30:
                self.senial = "work_in_progress"
            elif nombre_clase == "forward" and ancho >= 29:
                self.senial = "forward"
            elif nombre_clase == "traficlight_green" and ancho >= 10 and ancho <= 17:
                self.senial = "traficlight_green"
            elif nombre_clase == "traficlight_yellow" and ancho >= 10 and ancho <= 15:
                self.senial = "traficlight_yellow"
            elif nombre_clase == "traficlight_red" and ancho >= 10 and ancho <= 15:
                self.senial = "traficlight_red"

        # Publicar la señal detectada si es diferente a la última señal publicada o si es semáforo rojo
        if self.senial != self.last_detection_time or self.senial == "traficlight_red":
            self.current_senial = self.senial
            self.last_detection_time = self.senial
            self.signal_pub.publish(String(data=self.current_senial))  # Publicar la señal

        # Mostrar la imagen procesada
        cv2.imshow("DETECCION Y SEGMENTACION", image)
        if cv2.waitKey(1) == 27:  # Cerrar el programa si se presiona 'ESC'
            self.destroy_node()
            rclpy.shutdown()
    
    # Destructor de la clase
    def __del__(self):
        if not self.usar_camara_robot:
            self.cap.release()  # Liberar la captura de video
        cv2.destroyAllWindows()  # Cerrar todas las ventanas de OpenCV

# Función principal
def main(args=None): 
    rclpy.init(args=args)  # Inicializar ROS2
    usar_camara_robot = True  # Cambiar esto a True para usar la cámara del robot
    cv_e = Vision(usar_camara_robot)  # Crear instancia del nodo Vision
    rclpy.spin(cv_e)  # Mantener el nodo en funcionamiento
    cv_e.destroy_node()  # Destruir el nodo al finalizar
    rclpy.shutdown()  # Apagar ROS2

# Punto de entrada del script
if __name__ == '__main__': 
    main()
