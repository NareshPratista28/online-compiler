import java.util.Scanner;

public class HitungLuasSegitiga {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        System.out.print("Masukkan panjang sisi pertama: ");
        double a = scanner.nextDouble(); 
        
        System.out.print("Masukkan panjang sisi kedua: ");
        double b = scanner.nextDouble(); 
        double c = 0; 

        double luas, keliling;
        luas = Math.sqrt((a * b) / 4.0);
        
        keliling = a + b + c;

        System.out.println("---------------------------");
        System.out.println("Panjang sisi pertama: " + a);
        System.out.println("Panjang sisi kedua: " + b);
        System.out.println("Luas segitiga: " + luas);
        System.out.println("Keliling segitiga: " + keliling);

        scanner.close();
    }
}