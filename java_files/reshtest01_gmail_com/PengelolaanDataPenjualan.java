package reshtest01_gmail_com; 

public class PengelolaanDataPenjualan {
    public static void main(String[] args) {
        String kodeBarang = "Laptop";
        int jumlahItem = 2;
        double hargaBarang;
        double diskon;

        if (kodeBarang.equals("Laptop")) {
            hargaBarang = 500000;
        } else {
            hargaBarang = 0; // Harga default jika bukan laptop
        }

        double totalPembelian = jumlahItem * hargaBarang;

        if (totalPembelian > 100000) {
            diskon = 10;
        } else {
            diskon = 0;
        }

        double nilaiDiskon = (diskon / 100) * totalPembelian;
        double totalAkhir = totalPembelian - nilaiDiskon;

        System.out.println("Kode barang: " + kodeBarang);
        System.out.println("Jumlah item: " + jumlahItem);
        System.out.println("Harga barang: Rp" + hargaBarang);
        System.out.println("Diskon: " + diskon + "%");
        System.out.println("Total pembelian: Rp" + totalAkhir);
    }
}