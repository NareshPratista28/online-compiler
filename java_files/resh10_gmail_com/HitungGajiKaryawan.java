package resh10_gmail_com; 

public class HitungGajiKaryawan {
    public static void main(String[] args) {
        int jumlahJamKerja = 220; // Masukkan jumlah jam kerja
        double tarifPerJam = 50000.0; // Masukkan tarif per jam

        double totalGaji = jumlahJamKerja * tarifPerJam; // Perhitungan awal

        if (totalGaji > 10000000.0) { // Cek apakah totalGaji lebih besar dari Rp 10.000.000,00
            totalGaji += 500000.0; // Tambahkan biaya bonus sebesar Rp 500.000,00 jika kondisi terpenuhi
        }

        System.out.println("Detail Gaji Karyawan:");
        System.out.println("Jumlah Jam Kerja: " + jumlahJamKerja);
        System.out.println("Tarif Per Jam: Rp" + tarifPerJam);
        System.out.println("Total Gaji: Rp" + totalGaji);
    }

    // Tambahkan method untuk keperluan pengujian JUnit
    public static double hitungTotalGaji(int jam, double tarif) {
        double total = jam * tarif;
        if (total > 10000000.0) {
            total += 500000.0;
        }
        return total;
    }
}