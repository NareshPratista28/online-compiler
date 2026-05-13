package resh12_gmail_com; 

public class HitungVolumeBola {
    public static void main(String[] args) {
        double jariJari = 5; // Inisialisasi dengan 5
        
        double volumeBola; // Deklarasi variabel
        
        volumeBola = 4/3 * Math.PI * Math.pow(jariJari, 3); // Gunakan rumus volume bola
        
        System.out.println("Jari-jari: " + jariJari);
        System.out.println("Volume bola: " + volumeBola);
    }
}