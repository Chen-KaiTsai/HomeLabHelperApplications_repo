// Pin 27 -> out
// Pin 3v3 -> +
// Pin GND -> -

#include "main.h"

void setup() {
  Serial.begin(115200);
  while (!Serial)
    ;  // wait for Serial port to be opened
  Serial.printf("ESP32 Start\n");

  WiFi.mode(WIFI_STA);

  // Scan WiFi Networks
  netFunction::printWiFiScan();

  // Start Connect to WiFi
  WiFi.begin(ssid, pwd);

  while (WiFi.status() != WL_CONNECTED) {
    Serial.printf("Wait for WiFi network connection\n");
    delay(1000);
  }

  WiFi.printDiag(Serial);
  Serial.printf("IP Address : ");
  Serial.println(WiFi.localIP());

  server.on("/", netFunction::handleRoot);
  server.onNotFound(netFunction::handleNotFound);

  server.begin();

  Serial.printf("Start Temp Sensor Setup\n\n");

  // DHT22 : AM2302
  dht.setup(27, DHTesp::AM2302);

  // Sync Time with NTP
  configTime(utcOffset, daylightOffset, ntpServer);
  delay(1000);

  // Setup Sync Time with NTP regularly
  xTaskCreate(syncTimeWithFTP, "taskSyncTime", 1000, NULL, 1, NULL);
}

void loop() {
  server.handleClient();
}
