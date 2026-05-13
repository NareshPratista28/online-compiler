package resh12_gmail_com; 

public class HitungTotalBiayaProduksi {
    public static void main(String[] args) {
        int jumlahMesin = 10;
        
        double hargaMesin = 500000.0; // Inisialisasi dengan 500000.0
        double totalBiayaProduksi; // Deklarasi variabel

        totalBiayaProduksi = jumlahMesin * hargaMesin; // Gunakan hargaMesin

        System.out.println("Detail Biaya Produksi:");
        System.out.println("Jumlah mesin: " + jumlahMesin);
        System.out.println("Harga mesin: Rp" + hargaMesin);
        System.out.println("Total biaya produksi: Rp" + totalBiayaProduksi);
    } 
}