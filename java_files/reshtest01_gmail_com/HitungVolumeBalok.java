package reshtest01_gmail_com; 

public class HitungVolumeBalok {
    public static void main(String[] args) {
        // 1. Deklarasi variabel dengan nilai awal
        double panjang = 10.0; // Inisialisasi panjang dengan 10.0
        double lebar = 5.0;    // Inisialisasi lebar dengan 5.0
        double tinggi = 3.0;   // Inisialisasi tinggi dengan 3.0

        // 2. Implementasi rumus volume balok
        double volume = panjang * lebar * tinggi;

        // 3. Struktur kondisional
        if (volume > 0) {
            System.out.println("Volume balok positif!");
        } else if (volume < 0) {
            System.out.println("Volume balok negatif!");
        } else {
            System.out.println("Volume balok sama dengan nol!");
        }

        // Output detail
        System.out.println("Panjang: " + panjang);
        System.out.println("Lebar: " + lebar);
        System.out.println("Tinggi: " + tinggi);
        System.out.println("Volume balok: " + volume);
    }
}