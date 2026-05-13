import java.util.Scanner;

public class PenjualanBuku {
    public static void main(String[] args) {
        // 1. Deklarasi variabel
        double harga; 
        int stok; 
        int jumlahBeli; // Jumlah yang ingin dibeli user
        double totalPenjualan = 0; // Total uang yang didapat
        
        Scanner scanner = new Scanner(System.in);

        // Input data buku
        System.out.print("Masukkan harga buku: ");
        harga = scanner.nextDouble(); 

        System.out.print("Masukkan stok buku saat ini: ");
        stok = scanner.nextInt(); 

        // 2. Kondisi untuk menentukan apakah buku dijual atau tidak
        if (stok > 0) {
            System.out.println("Buku tersedia. Harga per buku: Rp" + harga);
            
            // 3. Menghitung total penjualan berdasarkan input jumlah buku yang terjual
            System.out.print("Masukkan jumlah buku yang terjual/dibeli: ");
            jumlahBeli = scanner.nextInt();

            // Validasi apakah stok mencukupi untuk jumlah yang dibeli
            if (jumlahBeli <= stok) {
                totalPenjualan = harga * jumlahBeli;
                System.out.println("Transaksi Berhasil!");
            } else {
                System.out.println("Transaksi Gagal! Stok tidak mencukupi.");
            }
            
        } else {
            System.out.println("Maaf, stok buku habis! Buku tidak dapat dijual.");
        }

        // Tampilkan hasil akhir
        System.out.println("-------------------------------");
        System.out.println("Laporan Penjualan Hari Ini:");
        System.out.println("Total Penjualan (Pendapatan): Rp" + totalPenjualan);
        
        scanner.close();
    }
}