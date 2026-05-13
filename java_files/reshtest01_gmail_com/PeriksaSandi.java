package reshtest01_gmail_com; 

import java.util.Scanner;

public class PeriksaSandi {
    public static void main(String[] args) {
        Scanner input = new Scanner(System.in);
        
        // Pastikan tidak ada teks prompt seperti "Masukkan sandi" jika sistem test otomatis
        if (!input.hasNextLine()) return;
        String kataSandi = input.nextLine(); 
        
        int jumlahHurufBesar = 0;
        int jumlahHurufKecil = 0;
        int jumlahAngka = 0;

        for (int i = 0; i < kataSandi.length(); i++) {
            char karakter = kataSandi.charAt(i);

            if (Character.isUpperCase(karakter)) {
                jumlahHurufBesar++;
            } else if (Character.isLowerCase(karakter)) {
                jumlahHurufKecil++;
            } else if (Character.isDigit(karakter)) {
                jumlahAngka++;
            }
        }

        // Output harus persis sesuai instruksi
        if (jumlahHurufBesar >= 1 && jumlahHurufKecil >= 1 && jumlahAngka >= 3) {
            System.out.println("Kata sandi aman!");
        } else {
            System.out.println("Kata sandi tidak aman!");
        }
        
        input.close();
    }
}