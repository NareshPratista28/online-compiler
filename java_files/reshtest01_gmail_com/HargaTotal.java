package reshtest01_gmail_com; 

import java.util.Scanner;

public class HargaTotal {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        System.out.print("Masukkan harga buku (float): ");
        float hargaBuku = scanner.nextFloat();
        
        scanner.nextLine(); // Membersihkan buffer

        System.out.print("Masukkan kategori buku (Fisika, Matematika, atau Bahasa Indonesia): ");
        String kategoriBuku = scanner.nextLine(); 
        
        System.out.print("Masukkan jumlah buku: ");
        int jumlahBuku = scanner.nextInt();
        
        scanner.close();

        // Logika IF-ELSE sesuai ekspektasi Test Case
        if (kategoriBuku.equalsIgnoreCase("Fisika") || kategoriBuku.equalsIgnoreCase("Matematika")) {
            if (jumlahBuku >= 5) {
                float diskon = hargaBuku * 0.1f; 
                // Test 2 mengharapkan format: "Harga total dengan diskon: Rp ..."
                System.out.println("Harga total dengan diskon: Rp " + (hargaBuku - diskon) * jumlahBuku);
            } else {
                // Test 4 & 1 mengharapkan format: "Harga total tanpa diskon: Rp ..."
                System.out.println("Harga total tanpa diskon: Rp " + (hargaBuku * jumlahBuku));
            }
        } else if (kategoriBuku.equalsIgnoreCase("Bahasa Indonesia")) {
            float diskon = hargaBuku * 0.05f; 
            // Test 3 mengharapkan format desimal panjang (bawaan float/double)
            System.out.println("Harga total dengan diskon: Rp " + (hargaBuku - diskon) * jumlahBuku);
        } else {
            // Test 5: Pesan harus tepat
            System.out.println("Kategori buku tidak valid.");
        }
    }
}