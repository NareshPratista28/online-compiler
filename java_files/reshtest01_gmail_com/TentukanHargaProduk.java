package reshtest01_gmail_com; 

public class TentukanHargaProduk {
    public static void main(String[] args) {
        // 1. Deklarasi dan inisialisasi variabel
        String kategori = "elektronik";
        int jumlahStok = 10;

        // 2. Menggunakan if-else if-else untuk menentukan harga
        if (kategori.equals("elektronik")) {
            
            // Nested IF: Jika stok banyak (>50), harga lebih murah
            if (jumlahStok > 50) { 
                System.out.println("Harga elektronik: Rp 100.000");
            } else {
                // Jika stok sedikit (<= 50)
                System.out.println("Harga elektronik: Rp 150.000");
            }

        } else if (kategori.equals("fashion")) {
            
            // Nested IF: Jika stok sedikit (<20), harga lebih mahal
            if (jumlahStok < 20) { 
                System.out.println("Harga fashion: Rp 200.000");
            } else {
                System.out.println("Harga fashion: Rp 150.000");
            }

        } else {
            // 3. Tampilkan harga standar untuk kategori selain di atas
            System.out.println("Harga untuk kategori lainnya: Rp 120.000");
        }
    }
}