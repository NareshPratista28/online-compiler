package resh12_gmail_com; 

public class HitungVolumeBalok {
    public static void main(String[] args) {
        double panjang = 10.0; // Inisialisasi panjang dengan 10.0
        double lebar = 5.0;   // Inisialisasi lebar dengan 5.0
        double tinggi = 3.0;  // Inisialisasi tinggi dengan 3.0

        double volume = panjang * lebar * tinggi; // Gunakan rumus: panjang * lebar * tinggi

        if (volume > 0) {
            System.out.println("Volume balok positif!");
        } else if (volume < 0) {
            System.out.println("Volume balok negatif!");
        } else {
            System.out.println("Volume balok sama dengan nol!");
        }

        // Tampilkan volume
        System.out.println("Panjang: " + panjang);
        System.out.println("Lebar: " + lebar);
        System.out.println("Tinggi: " + tinggi);
        System.out.println("Volume balok: " + volume);
    }
}