import paho.mqtt.client as mqtt
import threading


class LocalClient(threading.Thread):

    def __init__(self, client_id='people-counter', host='localhost', port=1883):
        threading.Thread.__init__(self)
        self.host = host
        self.port = port
        self.client_id = client_id
        self.client = mqtt.Client(client_id=self.client_id)

    def publish(self, topic, payload):
        self.client.publish(topic, payload)
        print('[MQTT_CLIENT] publish to ' + topic + ' payload: ' + payload)

    def run(self):
        print('[MQTT_CLIENT] connecting to mqtt -> ' + self.host + ':' + str(self.port))
        self.client.connect(self.host, self.port, 60)
        self.client.loop_forever()
