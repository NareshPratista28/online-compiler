package reshtest01_gmail_com; 

import java.util.Scanner;

public class LuasLingkaran {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        double jariJari = scanner.nextDouble(); // Membaca input dari pengguna

        if (jariJari >= 0.0) { // Periksa apakah jari-jari tidak negatif
            double luasLingkaran = Math.PI * jariJari * jariJari; // Menghitung luas lingkaran dengan rumus A = πr^2
            System.out.println("Luas lingkaran adalah: " + String.format("%.2f", luasLingkaran)); // Menampilkan hasil dengan dua desimal
        } else {
            System.out.println("Jari-jari tidak boleh negatif.");
        }
    }
}