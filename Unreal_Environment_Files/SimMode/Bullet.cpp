// Fill out your copyright notice in the Description page of Project Settings.


#include "Bullet.h"
#include "Components/StaticMeshComponent.h"
#include "Components/SphereComponent.h"
#include "GameFramework/ProjectileMovementComponent.h"
// #include "ConstructorHelpers.h"

#define COLLISION_OPT ECC_EngineTraceChannel2

// Sets default values
ABullet::ABullet()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

	BulletMesh = CreateDefaultSubobject<UStaticMeshComponent>("BulletMesh");
	SetRootComponent(BulletMesh);

	BulletMovement = CreateDefaultSubobject<UProjectileMovementComponent>("BulletMovement");
	BulletMovement->InitialSpeed = 1500.f;
	BulletMovement->MaxSpeed = 1500.f;
	BulletMovement->bRotationFollowsVelocity = true;

	// CollisionComponent = CreateDefaultSubobject<USphereComponent>(TEXT("SphereComponent"));
	// CollisionComponent->InitSphereRadius(1000.f);

	float randX = -1.0f;
	float randY = FMath::RandRange(-0.15f,0.15f);
	float randZ = FMath::RandRange(0.7f,1.1f);

	FVector ShootDirection = FVector(randX,randY,randZ);
	BulletMovement->Velocity = ShootDirection * BulletMovement->InitialSpeed;

	static ConstructorHelpers::FObjectFinder<UStaticMesh> MeshAsset(TEXT("/Game/StarterContent/Shapes/Shape_Sphere"));

	if(MeshAsset.Succeeded()){
		UStaticMesh* Asset = MeshAsset.Object;
    	BulletMesh->SetStaticMesh(Asset);
	}

	// BulletMesh->OnComponentBeginOverlap.AddDynamic(this, &ABullet::OnHit);
	// CollisionComponent->BodyInstance.SetCollisionProfileName(TEXT("Overlap"));
	// CollisionComponent->OnComponentHit.AddDynamic(this, &ABullet::OnHit);
	BulletMesh->SetCollisionProfileName(TEXT("Block")); // NoCollision Also works. A list of presets can be found or customizations can be made in collision menu in project settings
	// BulletMesh->SetCollisionObjectType(COLLISION_OPT);
}

// Called when the game starts or when spawned
void ABullet::BeginPlay()
{
	Super::BeginPlay();
	
}

// Called every frame
void ABullet::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

	curTick++;

	FVector NewLocation = GetActorLocation();
	FRotator NewRotation = GetActorRotation();

	SetActorLocationAndRotation(NewLocation, NewRotation);

	if(abs(this->BulletMovement->Velocity.X) < 1.0 && deleteTick == -1){
		deleteTick = curTick + 8;
	}

	if(deleteTick != -1){
		this->BulletMovement->Velocity.X = -15;
	}

	if(curTick == deleteTick){
		deleteTick = -1;
		curTick = 0;
		ABullet::OnHit();
	}
}

void ABullet::OnHit(){
	UE_LOG(LogTemp, Error, TEXT("It's a collision!"));
	this->Destroy();
}