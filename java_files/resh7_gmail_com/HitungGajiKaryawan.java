package resh7_gmail_com; 

public class HitungGajiKaryawan {
    public static void main(String[] args) {
        int jumlahJamKerja = 40; // Inisialisasi dengan 40
        
        double upahPerJam = 10000; // Inisialisasi nilai upah per jam
        double gaji; // Deklarasi variabel

        gaji = jumlahJamKerja * upahPerJam; // Gunakan upahPerJam

        System.out.println("Detail Gaji Karyawan:");
        System.out.println("Jumlah Jam Kerja: " + jumlahJamKerja);
        System.out.println("Upah per jam: Rp" + upahPerJam);
        System.out.println("Gaji karyawan: Rp" + gaji);
    }
}