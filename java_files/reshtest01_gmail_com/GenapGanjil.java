package reshtest01_gmail_com; 

import java.util.Scanner;

public class GenapGanjil {
    public static void main(String[] args) {
        // Inisialisasi Scanner untuk menerima input dari keyboard
        Scanner scanner = new Scanner(System.in);
        
        // 1. Deklarasikan variabel inputBilangan
        int inputBilangan;

        System.out.print("Masukkan sebuah bilangan: ");
        
        // 2. Gunakan Scanner untuk mengambil input angka dari pengguna
        inputBilangan = scanner.nextInt();

        // 3. Gunakan if-else untuk menentukan genap atau ganjil
        // Logika: Bilangan genap adalah bilangan yang habis dibagi 2 (sisa bagi 0)
        if (inputBilangan % 2 == 0) {
            System.out.println(inputBilangan + " adalah bilangan GENAP.");
        } else {
            System.out.println(inputBilangan + " adalah bilangan GANJIL.");
        }
        
        // Menutup scanner (good practice)
        scanner.close();
    }
}