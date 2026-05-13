package reshtest02_gmail_com; 

public class HitungJarakTempuh {
    public static void main(String[] args) {
        // 1. Deklarasi dan inisialisasi kecepatan rata-rata
        double kecepatanRata = 60.0; // Inisialisasi dengan 60.0

        // 2. Deklarasi dan inisialisasi waktu tempuh
        double waktuTempuh = 5.0; // Waktu tempuh dalam jam

        // 3. Hitung jarak tempuh
        double jarakTempuh;
        jarakTempuh = kecepatanRata * waktuTempuh;

        // 4. Tampilkan hasil
        System.out.println("Kendaraan berkecepatan rata-rata: " + kecepatanRata + " km/jam");
        System.out.println("Waktu tempuh: " + waktuTempuh + " jam");
        System.out.println("Jarak tempuh: " + jarakTempuh + " km");
    }
}