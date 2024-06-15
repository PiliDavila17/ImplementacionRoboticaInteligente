import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

# Definición de la clase LineFollower, que hereda de Node
class LineFollower(Node):
    def __init__(self):
        # Inicializa la clase base con el nombre del nodo
        super().__init__('line_follower_node')
        self.bridge = CvBridge()

        #### SUSCRIPTOR ####
        # Suscripción al tópico de imagen de la cámara
        self.sub = self.create_subscription(Image, 'video_source/raw', self.camera_callback, 10)
        # Suscripción al tópico de señales
        self.signs = self.create_subscription(String, 'segnal', self.segnal_callback, 10)

        ### PUBLICADOR ####
        # Publicador para la imagen procesada
        self.pub = self.create_publisher(Image, 'processed_img', 10)
        # Publicador para el comando de velocidad
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # VARIABLES ###
        self.vel = Twist()  # Variable para los comandos de velocidad
        self.image_received_flag = False  # Bandera para verificar si se ha recibido una imagen
        self.kp = 0.05  # Constante proporcional para la velocidad angular
        self.kd = 0.01  # Constante derivativa para la velocidad angular
        self.last_error = 0.0  # Error previo para el cálculo derivativo
        self.default_speed = 2.0  # Velocidad lineal predeterminada
        self.current_signal = None  # Señal actual recibida
        self.following_line = True  # Bandera para seguir la línea
        self.signal_timers = []  # Lista de temporizadores de señales
        self.previous_signal = None  # Señal previa recibida
        self.ignoring_traffic_signals = False  # Bandera para ignorar señales de tráfico
        self.middle_line_detected = False  # Bandera para detección de la línea central

        ### TEMPORIZADOR ###
        dt = 0.1  # Intervalo de tiempo del temporizador
        self.timer = self.create_timer(dt, self.timer_callback)  # Crear el temporizador
        self.get_logger().info('Line Follower Node started')

    # Callback para la suscripción a la cámara
    def camera_callback(self, msg):
        try:
            # Convertir el mensaje de imagen a una imagen OpenCV
            self.cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.image_received_flag = True  # Bandera para indicar que se recibió una imagen
        except Exception as e:
            self.get_logger().info(f'Failed to get image: {str(e)}')
    
    # Callback para la suscripción a las señales
    def segnal_callback(self, sign):
        # Verificar que la señal no sea un semáforo
        if sign.data != "traficlight_red" and sign.data != "traficlight_yellow" and sign.data != "traficlight_green":
            self.previous_signal = sign.data  # Guardar la señal previa
            self.current_signal = sign.data  # Actualizar la señal actual
            self.get_logger().info('Signal received: ' + self.current_signal)
            self.process_signal()  # Procesar la señal

    # Método para procesar las señales recibidas
    def process_signal(self):
        self.clear_signal_timers()  # Limpiar los temporizadores de señales previos

        if self.current_signal == "stop":
            self.get_logger().info('Stopping')
            self.following_line = False  # Dejar de seguir la línea
            self.vel.linear.x = 0.0
            self.vel.angular.z = 0.0
            self.cmd_vel_pub.publish(self.vel)  # Publicar el comando de velocidad
            self.signal_timers.append(self.create_timer(10.0, self.front))
        elif self.current_signal == "turn_right":
            self.get_logger().info('Turning right')
            self.vel.linear.x = 0.0
            self.vel.angular.z = 0.0
            self.cmd_vel_pub.publish(self.vel)
            self.vel.angular.z = 0.0
            self.vel.linear.x = 0.07  # Ajustar el valor según sea necesario
            self.cmd_vel_pub.publish(self.vel)
            self.following_line = False
            self.signal_timers.append(self.create_timer(1.0, self.right_t))
        elif self.current_signal == "turn_left":
            self.get_logger().info('Turning left')
            self.vel.linear.x = 0.0
            self.vel.angular.z = 0.0
            self.cmd_vel_pub.publish(self.vel)
            self.vel.angular.z = 0.0
            self.vel.linear.x = 0.07  # Ajustar el valor según sea necesario
            self.cmd_vel_pub.publish(self.vel)
            self.following_line = False
            self.signal_timers.append(self.create_timer(1.0, self.left_t))
        elif self.current_signal == "give_way":
            self.get_logger().info('Giving way')
            self.following_line = False
            self.vel.linear.x = 0.03  # Velocidad reducida
            self.vel.angular.z = 0.0
            self.cmd_vel_pub.publish(self.vel)
            self.signal_timers.append(self.create_timer(2.0, self.resume_line_following))
        elif self.current_signal == "work_in_progress":
            self.get_logger().info('Work in progress')
            self.following_line = False
            self.vel.linear.x = 0.03  # Velocidad reducida
            self.vel.angular.z = 0.0
            self.cmd_vel_pub.publish(self.vel)
            self.signal_timers.append(self.create_timer(5.0, self.resume_line_following))
        elif self.current_signal == "forward":
            self.get_logger().info('Moving forward')
            self.following_line = False
            self.vel.linear.x = 0.1  # Velocidad lineal máxima
            self.vel.angular.z = 0.0
            self.cmd_vel_pub.publish(self.vel)
            self.signal_timers.append(self.create_timer(4.0, self.resume_line_following))
        elif self.current_signal == "traficlight_green":
            self.get_logger().info('Traffic light green')
            self.following_line = False
            self.ignoring_traffic_signals = False  # Bandera para ignorar señales de tráfico
            if self.previous_signal:
                self.current_signal = self.previous_signal
                self.get_logger().info('Previous signal: ' + self.previous_signal)
                self.process_signal()
        elif self.current_signal == "traficlight_yellow":        
            self.get_logger().info('Traffic light yellow')
            self.following_line = False
            self.vel.linear.x = 0.02  # Velocidad reducida
            self.vel.angular.z = 0.0
            self.cmd_vel_pub.publish(self.vel)
            self.signal_timers.append(self.create_timer(0.5, self.resume_line_following))
        elif self.current_signal == "traficlight_red":
            self.get_logger().info('Traffic light red')
            self.following_line = False
            self.vel.linear.x = 0.0
            self.vel.angular.z = 0.0
            self.cmd_vel_pub.publish(self.vel)
            self.signal_timers.append(self.create_timer(1.0, self.resume_line_following))

    # Método para limpiar los temporizadores de señales
    def clear_signal_timers(self):
        for timer in self.signal_timers:
            timer.cancel()
        self.signal_timers = []

    # Método para reanudar el seguimiento de la línea
    def resume_line_following(self):
        self.ignoring_traffic_signals = True  # Bandera para ignorar señales de tráfico
        self.following_line = True
        self.current_signal = ""
        self.get_logger().info('Resuming line following')

    # Método para avanzar
    def front(self):
        self.vel.linear.x = 0.1
        self.vel.angular.z = 0.0
        self.cmd_vel_pub.publish(self.vel)
        self.get_logger().info('Going forward')
        self.signal_timers.append(self.create_timer(1.0, self.resume_line_following))

    # Método para avanzar (ignorando señales de tráfico)
    def move_forward(self):
        self.ignoring_traffic_signals = True  # Bandera para ignorar señales de tráfico
        self.vel.linear.x = 0.1
        self.vel.angular.z = 0.0
        self.cmd_vel_pub.publish(self.vel)
        self.get_logger().info('Going forward')
        self.signal_timers.append(self.create_timer(1.0, self.resume_line_following))

    # Método para avanzar (ignorando señales de tráfico)
    def move_forwardt(self):
        self.ignoring_traffic_signals = True  # Bandera para ignorar señales de tráfico
        self.vel.linear.x = 0.1
        self.vel.angular.z = 0.0
        self.cmd_vel_pub.publish(self.vel)
        self.get_logger().info('Going forward')
        self.signal_timers.append(self.create_timer(1.0, self.resume_line_following))

    # Método para girar a la izquierda
    def left_t(self):
        self.vel.linear.x = 0.0
        self.vel.angular.z = 0.4
        self.cmd_vel_pub.publish(self.vel)
        self.get_logger().info('Turning left')
        self.signal_timers.append(self.create_timer(3.0, self.move_forward))
    
    # Método para girar a la derecha
    def right_t(self):
        self.vel.linear.x = 0.0
        self.vel.angular.z = -0.4
        self.cmd_vel_pub.publish(self.vel)
        self.get_logger().info('Turning right')
        self.signal_timers.append(self.create_timer(3.0, self.move_forwardt))

    # Callback del temporizador
    def timer_callback(self):
        if self.image_received_flag and self.following_line:
            image = self.cv_img.copy()
            
            # Limitar la imagen solo a la parte inferior
            height, width, _ = image.shape
            roi_height = height // 2  # Usar la mitad inferior de la imagen
            roi = image[-roi_height:, :]

            # Crear una máscara para el color negro con umbrales ajustados para negro más claro
            lower_black = np.array([0, 0, 0])
            upper_black = np.array([180, 255, 80])  # Límite superior ajustado para detectar negro más claro
            mask = cv2.inRange(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV), lower_black, upper_black)

            # Definir las coordenadas de los quince rectángulos
            num_rects = 15
            rect_width = (width * 2) // (3 * num_rects)
            rect_height = roi_height // 3  # Reducir la altura de los rectángulos a un tercio del ROI

            # Centrar los rectángulos horizontalmente
            offset = (width - (rect_width * num_rects)) // 2

            # Ajustar la posición de altura de los rectángulos para estar más abajo
            rect_y = roi_height - rect_height - 10  # Mover los rectángulos para estar más cerca de la parte inferior

            # Crear una lista de rectángulos
            rects = [(i * rect_width + offset, rect_y, rect_width, rect_height) for i in range(num_rects)]

            # Dibujar los rectángulos en la imagen para visualización
            for rect in rects:
                cv2.rectangle(roi, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0), 2)

            detections = [False] * num_rects

            for i, rect in enumerate(rects):
                # Extraer el área del rectángulo de la máscara
                rect_mask = mask[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
                # Si hay suficientes píxeles negros en el área, considerarlo como detectado
                if cv2.countNonZero(rect_mask) > (rect[2] * rect[3] // 2):  # Umbral de ejemplo
                    detections[i] = True

            # Determinar el movimiento basado en las detecciones
            center_index = num_rects // 2
            if detections[center_index]:
                self.vel.angular.z = 0.0
                self.vel.linear.x = 0.2  # Velocidad lineal máxima
                self.middle_line_detected = True
                self.last_error = 0.0  # Reiniciar el último error
            elif self.middle_line_detected:
                deviation = sum([(i - center_index) * detections[i] for i in range(num_rects)])
                error = -self.kp * deviation
                derivative = self.kd * (error - self.last_error) / 0.1  # dt es 0.1
                self.last_error = error
                self.vel.angular.z = max(-1.0, min(1.0, error + derivative))
                self.vel.linear.x = max(0.05, 0.2 - 0.1 * abs(deviation))  # Disminuir la velocidad lineal a medida que aumenta la desviación
            else:
                self.vel.angular.z = 0.0
                self.vel.linear.x = 0.0

            # Comprobar si todas las detecciones son negras
            if all(detections):
                self.vel.linear.x = 0.0
                self.vel.angular.z = 0.0
                self.cmd_vel_pub.publish(self.vel)

            for i, rect in enumerate(rects):
                color = (0, 255, 0) if detections[i] else (255, 0, 0)
                cv2.rectangle(roi, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), color, -1 if detections[i] else 2)

            self.cmd_vel_pub.publish(self.vel)
            self.pub.publish(self.bridge.cv2_to_imgmsg(image, 'bgr8'))

# Método principal
def main(args=None):
    rclpy.init(args=args)  # Inicializar ROS2
    node = LineFollower()  # Crear instancia del nodo LineFollower
    rclpy.spin(node)  # Mantener el nodo en funcionamiento
    node.destroy_node()  # Destruir el nodo al finalizar
    rclpy.shutdown()  # Apagar ROS2

# Punto de entrada del script
if __name__ == '__main__':
    main()
