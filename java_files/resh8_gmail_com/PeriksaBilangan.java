package resh8_gmail_com; 

public class PeriksaBilangan {
    public static void main(String[] args) {
        int bilangan = 5; // Masukkan nilai yang ingin diperiksa
        
        if (bilangan > 0) { 
            System.out.println("Bilangan adalah positif.");
        } else if (bilangan < 0) { 
            System.out.println("Bilangan adalah negatif.");
        } else {
            System.out.println("Bilangan adalah nol.");
        }
    }
}