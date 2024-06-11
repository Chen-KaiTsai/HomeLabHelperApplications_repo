#include <WiFi.h>
#include <WebServer.h>
#include <DHTesp.h>

DHTesp dht;
WebServer server(80);
const char ssid[] = "";
const char pwd[] = "";
const char ntpServer[] = "pool.ntp.org";
const uint16_t utcOffset = 28800; // UTC+8 offset
const uint8_t daylightOffset = 0; // day light saving offset

namespace netFunction {
void printWiFiScan() {
  // Start probing for WiFi
  int numberNetwork = WiFi.scanNetworks();

  char delimiter[40] = "===================================";

  Serial.printf("%s\n\n", delimiter);

  if (numberNetwork == 0) {
    Serial.printf("No network found\n\n");
  } else {
    Serial.printf("%d network found\nStart listing networks\n\n", numberNetwork);
    for (int i = 0; i < numberNetwork; ++i) {
      Serial.printf("%d SSID : ", i);
      Serial.print(WiFi.SSID(i));
      Serial.printf(" RSSI : ");
      Serial.print(WiFi.RSSI(i));
      Serial.printf("\n\n");
    }
  }
}

void handleRoot() {
  // Read and Update Sensor Data when website is accessed.
  struct tm now;
  if (!getLocalTime(&now)) {
    Serial.printf("Cannot get time\n");
    return;
  }
  
  uint tyear = now.tm_year + 1900;
  uint tmonth = now.tm_mon + 1;
  uint tday = now.tm_mday;
  uint thour = now.tm_hour;
  uint tmin = now.tm_min;
  uint tsec = now.tm_sec;

  TempAndHumidity data = dht.getTempAndHumidity();
  Serial.printf("Temparature : %lf Humidity : %lf ", data.temperature, data.humidity);
  Serial.println(&now, "%Y/%m/%d %H:%M:%S");

  String tempString = String(data.temperature);
  String humidString = String(data.humidity);

  String yearString = String(tyear);
  String monthString = String(tmonth);
  String dayString = String(tday);
  String hourString = String(thour);
  String minString = String(tmin);
  String secString = String(tsec);

  String HTML = "\
  <!DOCTYPE html>\
  <html><head><meta charset='utf-8'></head>\
  <body>Temperature : \
  " + tempString + " Humidity : " + humidString + 
  "<br>Current Time : " + yearString + "/" +
  monthString + "/" + dayString + " " + 
  hourString + ":" + minString + ":" + secString +
  "</body></html>\
  ";

  server.send(200, "text/html", HTML);
}

void handleNotFound() {
  String plainText = "File Not found\n";
  server.send(404, "text/plain", plainText);
}
};

void syncTimeWithFTP(void* pvParam) {
  while (true) {
    Serial.println("Sync With FTP\n");
    configTime(utcOffset, daylightOffset, ntpServer);
    uint resyncMS = 7200 * 1000;
    vTaskDelay(resyncMS / portTICK_PERIOD_MS);
  }
}
