package reshtest01_gmail_com; 

import java.util.Scanner;

public class BelanjaOnline {
    public static void main(String[] args) {
        Scanner input = new Scanner(System.in);
        double harga = 0;
        int jumlah = 0;
        String diskon = "";
        
        System.out.print("Masukkan harga produk: ");
        harga = input.nextDouble();
        
        System.out.print("Masukkan jumlah produk yang dibeli: ");
        jumlah = input.nextInt();
        
        // Menentukan status diskon
        if (jumlah > 10) {
            diskon = "Diskon 10%";
        } else {
            diskon = "Tidak ada diskon";
        }
        
        double totalHarga;
        // Menghitung total harga berdasarkan status diskon
        if (diskon.equals("Diskon 10%")) {
            totalHarga = (harga * jumlah) * 0.9;
        } else {
            totalHarga = harga * jumlah;
        }
        
        // Menampilkan output sesuai format yang diminta sistem test:
        // 'Tidak ada diskon Rp 75.0' atau 'Diskon 10% Rp 198.0'
        System.out.println(diskon + " Rp " + totalHarga);
        
        input.close();
    }
}